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
use pounce_common::types::{Index, Number};
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
    /// A branch NLP: each pair with one side held at zero, the other kept
    /// non-negative — run from a reported point that is not B-stationary,
    /// along the branch its B-stationarity check found a descent on
    /// ([`check_b_stationarity`]). Not in [`STAGES`].
    Branch,
}

impl MpccStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact product",
            Self::Continuation => "Scholtes continuation",
            Self::Penalty => "l1 penalty",
            Self::Branch => "branch NLP",
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

/// Which point the driver reported, and on what grounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reported {
    /// A stage answered: converged, products in tolerance, certified S (or
    /// not determined).
    Answer(MpccStage),
    /// No stage certified S; this stage's W-stationary point is reported.
    WeakFallback(MpccStage),
    /// No stage certified any stationarity; this stage's converged point with
    /// every product in tolerance is reported.
    FeasibleUncertified(MpccStage),
    /// No stage produced a feasible point; the exact-product stage's own
    /// verdict (or, if it was refused with no point, the first stage that
    /// has one) is reported.
    Verdict,
}

/// One line for the console: what answered, and what the answer is.
pub fn describe_outcome(
    reported: Reported,
    n_solves: usize,
    total_iterations: usize,
    max_abs_product: Number,
    certificate: Option<&Certificate>,
) -> String {
    let solves = format!(
        "{n_solves} lowered solve{}, {total_iterations} iterations in all",
        if n_solves == 1 { "" } else { "s" }
    );
    let head = match reported {
        Reported::Answer(stage) => format!(
            "pounce: complementarity: answered by the {} formulation ({solves}, max |G·H| = \
             {max_abs_product:.1e})",
            stage.label(),
        ),
        Reported::WeakFallback(stage) => format!(
            "pounce: complementarity: no formulation produced a strongly stationary answer \
             ({solves}); reporting the {} formulation's weakly stationary point (max |G·H| = \
             {max_abs_product:.1e})",
            stage.label(),
        ),
        Reported::FeasibleUncertified(stage) => format!(
            "pounce: complementarity: no formulation produced a stationary answer ({solves}); \
             reporting the {} formulation's feasible point (max |G·H| = {max_abs_product:.1e})",
            stage.label(),
        ),
        Reported::Verdict => format!(
            "pounce: complementarity: no formulation produced an answer ({solves}); reporting \
             the exact-product solve's verdict"
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
                StationarityClass::None if !c.feasible => {
                    return format!(
                        "{head}; {} pair{}, {} biactive; stationarity: none (the point violates a \
                         row or bound by {:.1e}, beyond constr_viol_tol).",
                        c.n_pairs,
                        if c.n_pairs == 1 { "" } else { "s" },
                        c.n_biactive,
                        c.primal_violation
                    );
                }
                StationarityClass::None => "none (not a stationary point of the model)",
                StationarityClass::NotSearched => {
                    "not determined (the model or its multiplier set is too large to certify)"
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

/// Which formulations a solve runs, from `complementarity_strategy`. The
/// penalty runs under every strategy, as the fallback when nothing answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Both,
    Exact,
    Continuation,
}

impl Strategy {
    /// The option's value; `None` for anything else (the registry rejects
    /// other values before a solve starts).
    pub fn parse(s: &str) -> Option<Strategy> {
        match s {
            "both" => Some(Strategy::Both),
            "exact" => Some(Strategy::Exact),
            "continuation" => Some(Strategy::Continuation),
            _ => None,
        }
    }

    pub fn runs(self, stage: MpccStage) -> bool {
        !matches!(
            (self, stage),
            (Strategy::Exact, MpccStage::Continuation) | (Strategy::Continuation, MpccStage::Exact)
        )
    }
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
    /// Not determined; nothing is claimed. The multipliers are not unique
    /// and the set is too large to search: more than [`MAX_DENSE_ENTRIES`]
    /// entries in the system, or more than [`MAX_SEARCHED_COLUMNS`]
    /// sign-constrained columns. Also reported if the sparse factorization
    /// fails outright.
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
    /// The band `residual` is judged against: `max(1e-6, tol)` plus the
    /// `sqrt(tol)` sign band, both times the dual scale (see [`certify`]).
    pub resid_tol: Number,
    /// The residual is within the floor alone,
    /// `max(1e-6, tol)` times the dual scale. An `S` point that is not
    /// `tight` reproduces `∇f` only to the `sqrt(tol)` band: the driver keeps
    /// it and reports it only if no formulation produces a tight one. With a
    /// multiplier set, the test is on the sign-free fit, as the old gate
    /// into the set search was; the search itself then takes the band.
    pub tight: bool,
    /// Largest violation of a row or variable bound at the point (pair
    /// sides included), in the model's units.
    pub primal_violation: Number,
    /// `primal_violation` is within [`CertificateInput::feas_tol`]. When it
    /// is not, the class is `None` whatever the multipliers say.
    pub feasible: bool,
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
    /// The feasibility tolerance (`constr_viol_tol`): a point violating a
    /// row or bound by more is not certified, stationary or not.
    pub feas_tol: Number,
}

/// Above this many sign-constrained columns, a rank-deficient system is not
/// searched (each NNLS step is an SVD of the passive columns).
pub const MAX_SEARCHED_COLUMNS: usize = 600;

/// Above this many entries in the `n × k` stationarity matrix, a system whose
/// multipliers are **not unique** is not searched and reports
/// [`StationarityClass::NotSearched`] (which [`answers`] accepts, so the
/// ladder is unchanged). The set search factors the matrix densely; at
/// `n ≈ 10⁴` with `k ≈ n + m` that is about 1 GB and 10¹² flops per call, a
/// size contact-implicit trajectory models reach. 4·10⁶ entries (32 MB)
/// keeps the call well under a second. A system with unique multipliers —
/// the common case, MPCC-LICQ — is certified at any size, sparsely (see
/// [`DENSE_ENTRIES`]).
pub const MAX_DENSE_ENTRIES: usize = 4_000_000;

/// Up to this many entries the stationarity system is factored densely (a
/// thin SVD, rank by LAPACK's threshold); above it, by the interior-point
/// method's own sparse LDLᵀ of the least-squares augmented system
/// ([`sparse_least_squares`]). The dense SVD's cost grows as `n·k²`: on a
/// 650-pair chain model (`n = 1950`) the solve took 4.8 s with the dense
/// certificate and 0.51 s with the sparse one, and at 1100 pairs 31 s
/// against 0.93 s, same class and answer. Below 2.5·10⁵ entries the SVD costs little and keeps the path
/// the certificate was validated on against `stationarity.py` (the Gate 0
/// corpus, bunny-hop, ball-in-cup); the CALIPSO rocket (`n = 856`) is past
/// it and certifies S on the sparse path exactly as it did densely. The
/// unit tests hold the two paths to the same class on every point they
/// check.
pub const DENSE_ENTRIES: usize = 250_000;

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
/// [`MAX_SEARCHED_COLUMNS`] and [`MAX_DENSE_ENTRIES`]; a larger set reports
/// `NotSearched`. Above [`DENSE_ENTRIES`] the unique multipliers come from a
/// sparse factorization rather than the SVD.
pub fn certify(inp: &CertificateInput<'_>) -> Certificate {
    certify_routed(inp, None)
}

/// Each pair's orientation, values and activity at a point, as both the
/// certificate and the B-stationarity check read them.
struct PairActivity {
    mixed: Vec<bool>,
    /// `(s_G, b)` and `(s_H, c)`: `G = s_G (x_j - b)`, `H = s_H (g_i - c)`.
    g_side: Vec<(Number, Number)>,
    h_side: Vec<(Number, Number)>,
    hv: Vec<Number>,
    g_act: Vec<bool>,
    h_act: Vec<bool>,
}

fn pair_activity(inp: &CertificateInput<'_>, corner: Number) -> PairActivity {
    // Pair sides and their family scales.
    // Each side's orientation from its one finite bound, as MpccTnlp reads
    // it: (+1, lower) or (-1, upper).
    let side = |lo: Number, hi: Number| -> (Number, Number) {
        if lo > -INF { (1.0, lo) } else { (-1.0, hi) }
    };
    // A mixed pair (`l <= x <= u ⟂ g`, the row free) is, at a given point,
    // the one-sided pair of the bound `x` is nearer: `x - l ⟂ g` at `l`,
    // `u - x ⟂ -g` at `u` (strictly between, `G` is inactive either way and
    // `g = 0` is the condition).
    let mixed: Vec<bool> = inp
        .pairs
        .iter()
        .map(|p| {
            pounce_nlp::mpcc::is_mixed(
                inp.x_l[p.var],
                inp.x_u[p.var],
                inp.g_l[p.row],
                inp.g_u[p.row],
            )
        })
        .collect();
    let g_side: Vec<(Number, Number)> = inp
        .pairs
        .iter()
        .zip(&mixed)
        .map(|(p, &mx)| {
            let (lo, hi) = (inp.x_l[p.var], inp.x_u[p.var]);
            if mx {
                if inp.x[p.var] - lo <= hi - inp.x[p.var] {
                    (1.0, lo)
                } else {
                    (-1.0, hi)
                }
            } else if lo > -INF && hi < INF {
                // A boxed variable in a one-sided pair: the row's finite side
                // names the bound in the pair, as MpccTnlp reads it; the
                // other bound is an ordinary limit (columns below).
                if inp.g_l[p.row] > -INF {
                    (1.0, lo)
                } else {
                    (-1.0, hi)
                }
            } else {
                side(lo, hi)
            }
        })
        .collect();
    let h_side: Vec<(Number, Number)> = inp
        .pairs
        .iter()
        .zip(&mixed)
        .zip(&g_side)
        .map(|((p, &mx), &(sg, _))| {
            if mx {
                (sg, 0.0)
            } else {
                side(inp.g_l[p.row], inp.g_u[p.row])
            }
        })
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
    PairActivity {
        mixed,
        g_side,
        h_side,
        hv,
        g_act,
        h_act,
    }
}

/// [`certify`], with the dense/sparse choice forced when `force_sparse` is
/// set — for holding the two paths to each other in tests.
fn certify_routed(inp: &CertificateInput<'_>, force_sparse: Option<bool>) -> Certificate {
    let n = inp.x.len();
    let m = inp.g.len();
    let corner = inp.solver_tol.max(0.0).sqrt();
    let row_tol = corner;

    let PairActivity {
        mixed,
        g_side,
        h_side,
        hv,
        g_act,
        h_act,
    } = pair_activity(inp, corner);
    let n_biactive = g_act.iter().zip(&h_act).filter(|(a, b)| **a && **b).count();
    let n_neither = g_act
        .iter()
        .zip(&h_act)
        .filter(|(a, b)| !**a && !**b)
        .count();

    // Primal feasibility, in the model's units: every row and every bound,
    // pair sides included. A point outside its own constraints is not a
    // stationary point of the model, whatever multipliers reproduce `∇f`
    // (an infeasibility verdict's point did read S before this check).
    let viol = |v: Number, lo: Number, hi: Number| {
        let below = if lo > -INF { lo - v } else { 0.0 };
        let above = if hi < INF { v - hi } else { 0.0 };
        below.max(above).max(0.0)
    };
    // A mixed pair's row is free, so no row bound carries the sign of `g` at
    // an active bound (`g >= 0` at `l`, `g <= 0` at `u`): measure it here.
    let mixed_sign = (0..inp.pairs.len())
        .filter(|&k| mixed[k] && g_act[k])
        .map(|k| (-hv[k]).max(0.0));
    let primal_violation = (0..m)
        .map(|i| viol(inp.g[i], inp.g_l[i], inp.g_u[i]))
        .chain((0..n).map(|j| viol(inp.x[j], inp.x_l[j], inp.x_u[j])))
        .chain(mixed_sign)
        .fold(0.0, Number::max);
    let feasible = primal_violation <= inp.feas_tol;

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
            // A one-sided pair variable's other bound, when it has one, is an
            // ordinary limit: a sign-constrained column when active. (A mixed
            // pair's other bound is the other branch, handled above.)
            if !mixed[k] {
                let (lo, hi) = (inp.x_l[j], inp.x_u[j]);
                let other_active = if sg > 0.0 {
                    hi < INF && hi - inp.x[j] <= row_tol
                } else {
                    lo > -INF && inp.x[j] - lo <= row_tol
                };
                if other_active {
                    cols.push((vec![(j, -sg)], Sign::NonNeg));
                }
            }
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
    let col_max = cols
        .iter()
        .flat_map(|(ents, _)| ents)
        .fold(0.0_f64, |m, &(_, v)| m.max(v.abs()));
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
    // One acceptance band on every path: the residual's own floor plus the
    // sign band. The point is `sqrt(tol)`-accurate near the boundary, so
    // `∇f` is reproduced only to that order; the set search always allowed
    // it, and the unique path holding the same point to `resid_tol` alone
    // rejected optimal IPM answers (MacMPEC `flp4-3`, `flp4-4`,
    // `portfl-i-1`: residuals 1.7e-6 to 2.1e-5 at f within 0.2% of f*).
    let accept_tol = resid_tol + sign_tol;
    let signs: Vec<Sign> = cols.iter().map(|(_, s)| *s).collect();

    let mut cert = Certificate {
        class: StationarityClass::None,
        n_pairs: inp.pairs.len(),
        n_biactive,
        n_neither,
        multipliers_unique: true,
        residual: Number::INFINITY,
        resid_tol: accept_tol,
        tight: false,
        primal_violation,
        feasible,
        worst_biactive_sign: 0.0,
    };
    if n_neither > 0 || !feasible {
        // Complementarity, or feasibility itself, fails at this point: no
        // class applies.
        return cert;
    }

    // Unique multipliers: the class is a sign check on them.
    let classify_unique = |cert: &mut Certificate, theta: &[Number]| {
        let ordinary_ok = signs
            .iter()
            .zip(theta)
            .all(|(s, t)| *s != Sign::NonNeg || *t >= -sign_tol);
        if !ordinary_ok {
            return;
        }
        let worst = signs
            .iter()
            .zip(theta)
            .filter(|(s, _)| **s == Sign::Biactive)
            .fold(0.0_f64, |w, (_, t)| w.min(*t));
        cert.worst_biactive_sign = worst;
        cert.class = if worst >= -sign_tol {
            StationarityClass::S
        } else {
            StationarityClass::W
        };
    };

    let sparse = force_sparse.unwrap_or(n.saturating_mul(k) > DENSE_ENTRIES);
    if sparse {
        let entries: Vec<Vec<(usize, Number)>> = cols.iter().map(|(e, _)| e.clone()).collect();
        match sparse_least_squares(n, &entries, &b) {
            LeastSquares::Solved(theta) => {
                let mut r = b.clone();
                for (ents, t) in entries.iter().zip(&theta) {
                    for &(row, v) in ents {
                        r[row] -= v * t;
                    }
                }
                cert.residual = r.iter().fold(0.0, |m: Number, v| m.max(v.abs()));
                cert.tight = cert.residual <= resid_tol;
                if cert.residual <= accept_tol {
                    classify_unique(&mut cert, &theta);
                }
                return cert;
            }
            LeastSquares::RankDeficient if n.saturating_mul(k) <= MAX_DENSE_ENTRIES => {
                // The set search below is dense; it fits.
            }
            LeastSquares::RankDeficient | LeastSquares::Failed => {
                cert.multipliers_unique = false;
                cert.residual = Number::NAN;
                cert.class = StationarityClass::NotSearched;
                cert.tight = true; // nothing measured looser: accepted as before
                return cert;
            }
        }
    }

    let mut a = Mat::<Number>::from_fn(n, k, |_, _| 0.0);
    for (c, (ents, _)) in cols.iter().enumerate() {
        for &(r, v) in ents {
            a[(r, c)] += v;
        }
    }

    // Sign-free least squares: the W test's lower bound on the residual,
    // and the unique multipliers when the columns have full rank.
    let (theta, rank) = lstsq(&a, &b);
    let r_free = resid_inf(&a, &theta, &b);
    cert.residual = r_free;
    cert.tight = r_free <= resid_tol;
    if r_free > accept_tol {
        return cert; // not stationary for any multipliers, signs aside
    }
    if rank == k {
        classify_unique(&mut cert, &theta);
        return cert;
    }

    // Non-unique multipliers: search the set.
    cert.multipliers_unique = false;
    let constrained = signs.iter().filter(|s| **s != Sign::Free).count();
    if constrained > MAX_SEARCHED_COLUMNS {
        cert.class = StationarityClass::NotSearched;
        cert.tight = true; // nothing measured looser: accepted as before
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
        if r <= accept_tol {
            cert.class = class;
            cert.residual = r;
            // Tight as the sign-free fit was: before the band was unified, a
            // point reached this search only with `r_free <= resid_tol`, and
            // then took the whole band (see
            // `the_set_search_band_is_resid_tol_plus_the_sign_band`). Those
            // points stay answers; one the old gate turned away is kept.
            cert.tight = r_free <= resid_tol;
            return cert;
        }
    }
    cert
}

/// What [`sparse_least_squares`] found.
#[derive(Debug, Clone, PartialEq)]
enum LeastSquares {
    /// The unique minimizer `θ`.
    Solved(Vec<Number>),
    /// The columns are (numerically) dependent: the minimizers form a set.
    RankDeficient,
    /// Malformed input, or the factorization or solve failed.
    Failed,
}

/// `min ‖A θ − b‖₂` for a sparse `A` (`n × k`, columns as `(row, value)`
/// lists, repeated rows summed), through the least-squares augmented system
///
/// ```text
/// [ s·I  A ] [ r ]   [ b ]
/// [ Aᵀ   0 ] [ θ ] = [ 0 ]      (s·r = b − A θ,  Aᵀ r = 0)
/// ```
///
/// factored by the same symmetric indefinite solver the interior-point
/// method uses on its own KKT systems (FERAL, through
/// [`SparseSymLinearSolverInterface`]), with `s = max(1, max|A|)`. The
/// system has exactly `n` positive and `k` negative eigenvalues when `A`
/// has full column rank, so the factorization's inertia is the rank test:
/// a zero pivot or any other negative count is [`LeastSquares::RankDeficient`],
/// and no answer is returned rather than a perturbed one.
fn sparse_least_squares(n: usize, cols: &[Vec<(usize, Number)>], b: &[Number]) -> LeastSquares {
    use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};

    let k = cols.len();
    if b.len() != n || cols.iter().flatten().any(|&(r, _)| r >= n) {
        return LeastSquares::Failed;
    }
    if k == 0 {
        return LeastSquares::Solved(Vec::new());
    }
    if k > n {
        return LeastSquares::RankDeficient;
    }
    let a_max = cols
        .iter()
        .flatten()
        .fold(0.0_f64, |m, &(_, v)| m.max(v.abs()));
    let s = a_max.max(1.0);
    // Lower triangle, 1-based (FERAL's triplet convention): the identity
    // block on the diagonal, Aᵀ below it in rows n+1..n+k.
    let mut trip: Vec<(Index, Index, Number)> = (1..=n as Index).map(|i| (i, i, s)).collect();
    for (j, c) in cols.iter().enumerate() {
        let mut c = c.clone();
        c.sort_by_key(|&(r, _)| r);
        for (r, v) in c {
            let (row, col) = ((n + j + 1) as Index, (r + 1) as Index);
            match trip.last_mut() {
                Some(last) if last.0 == row && last.1 == col => last.2 += v,
                _ => trip.push((row, col, v)),
            }
        }
    }
    let irn: Vec<Index> = trip.iter().map(|t| t.0).collect();
    let jcn: Vec<Index> = trip.iter().map(|t| t.1).collect();
    let mut solver = pounce_feral::FeralSolverInterface::new();
    let dim = (n + k) as Index;
    if solver.initialize_structure(dim, irn.len() as Index, &irn, &jcn) != ESymSolverStatus::Success
    {
        return LeastSquares::Failed;
    }
    for (slot, t) in solver.values_array_mut().iter_mut().zip(&trip) {
        *slot = t.2;
    }
    let mut rhs0 = vec![0.0; n + k];
    rhs0[..n].copy_from_slice(b);
    let mut z = rhs0.clone();
    match solver.multi_solve(true, &irn, &jcn, 1, &mut z, true, k as Index) {
        ESymSolverStatus::Success => {}
        ESymSolverStatus::Singular | ESymSolverStatus::WrongInertia => {
            return LeastSquares::RankDeficient;
        }
        _ => return LeastSquares::Failed,
    }
    // Iterative refinement on the same factor: the multipliers' signs are
    // what the class is read from, and the augmented system is as
    // ill-conditioned as A's columns are close to dependent.
    for _ in 0..REFINEMENT_STEPS {
        let mut corr = rhs0.clone();
        for &(r, c, v) in &trip {
            let (r, c) = ((r - 1) as usize, (c - 1) as usize);
            corr[r] -= v * z[c];
            if r != c {
                corr[c] -= v * z[r];
            }
        }
        if solver.multi_solve(false, &irn, &jcn, 1, &mut corr, false, 0)
            != ESymSolverStatus::Success
        {
            return LeastSquares::Failed;
        }
        z.iter_mut().zip(&corr).for_each(|(zi, ci)| *zi += ci);
    }
    if z.iter().all(|v| v.is_finite()) {
        LeastSquares::Solved(z[n..].to_vec())
    } else {
        LeastSquares::Failed
    }
}

const REFINEMENT_STEPS: usize = 2;

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
    // A column freed and dropped again in the same step (its least-squares
    // coefficient is not positive, so `alpha = 0`) would be picked again
    // next step with nothing changed; it sits out until some other column
    // enters and moves `x`.
    let mut blocked = vec![false; k];
    for _outer in 0..3 * k + 10 {
        let w = grad(&resid(&x));
        let Some((t, wt)) = (0..k)
            .filter(|&j| !passive[j] && !blocked[j])
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
        if passive[t] {
            blocked.fill(false);
        } else {
            blocked[t] = true;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// B-stationarity: whether a feasible direction along some branch descends.
// ---------------------------------------------------------------------------

/// Which side of a pair is held at zero along a branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeldSide {
    G,
    H,
}

/// What [`check_b_stationarity`] found.
#[derive(Debug, Clone, PartialEq)]
pub enum BVerdict {
    /// No branch has a first-order descent direction: B-stationary.
    B,
    /// The branch `held` (one entry per pair) descends: `∇f·d = decrease`
    /// (negative) for a step `d` in the unit box.
    NotB {
        held: Vec<HeldSide>,
        decrease: Number,
    },
    /// Not decided, with the reason.
    NotChecked(&'static str),
}

/// [`check_b_stationarity`]'s verdict and what it took.
#[derive(Debug, Clone, PartialEq)]
pub struct BCheck {
    pub verdict: BVerdict,
    pub n_biactive: usize,
    /// Branch LPs solved.
    pub branches_solved: usize,
    /// Whether every branch was enumerated (`false`: the heuristic above
    /// [`MAX_ENUMERATED_BIACTIVE`], which can find a descent but never
    /// certify B).
    pub enumerated: bool,
}

/// Up to this many biactive pairs every one of the `2^b` branches is
/// checked; above it, one heuristic branch.
pub const MAX_ENUMERATED_BIACTIVE: usize = 10;

/// Whether `x` is B-stationary: no direction `d` feasible for the linearized
/// constraints along any branch of the biactive pairs has `∇f·d < 0`.
///
/// Each branch is an LP: `min ∇f·d` over the linearized active rows and
/// bounds, with each pair's active side held (`∇G·d = 0` or `∇H·d = 0`) and,
/// on a biactive pair, the side the branch releases kept non-negative, in the
/// box `-1 <= d <= 1`. `d = 0` is always feasible, so every LP has optimum
/// `<= 0`; the point is B-stationary exactly when all of them are `0`. Up to
/// [`MAX_ENUMERATED_BIACTIVE`] biactive pairs all `2^b` branches are solved
/// (the LPEC by enumeration, no MILP); above it, the relaxed LP — both sides
/// of every biactive pair only non-negative — picks one branch (hold the side
/// its direction moves less), which is solved.
///
/// A decrease counts only below `-sqrt(solver_tol)·max(1, ‖∇f‖∞)`, the
/// certificate's sign band: near a biactive pair the point itself is only
/// `sqrt(tol)`-accurate. S implies B, so a caller checks only points the
/// certificate did not call S. The point must be feasible and every pair
/// complementary.
pub fn check_b_stationarity(inp: &CertificateInput<'_>) -> BCheck {
    let corner = inp.solver_tol.max(0.0).sqrt();
    let act = pair_activity(inp, corner);
    let np = inp.pairs.len();
    let biactive: Vec<usize> = (0..np).filter(|&k| act.g_act[k] && act.h_act[k]).collect();
    let mut out = BCheck {
        verdict: BVerdict::NotChecked(""),
        n_biactive: biactive.len(),
        branches_solved: 0,
        enumerated: false,
    };
    if (0..np).any(|k| !act.g_act[k] && !act.h_act[k]) {
        out.verdict = BVerdict::NotChecked("a pair is not complementary at the point");
        return out;
    }
    let scale = inp.grad_f.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let band = corner * scale;
    // Held side per pair, `None` on a biactive pair left to the branch.
    let held_of = |choice: &dyn Fn(usize) -> Option<HeldSide>| -> Vec<Option<HeldSide>> {
        (0..np)
            .map(|k| match (act.g_act[k], act.h_act[k]) {
                (true, false) => Some(HeldSide::G),
                (false, true) => Some(HeldSide::H),
                _ => choice(k),
            })
            .collect()
    };
    let full = |h: &[Option<HeldSide>]| -> Vec<HeldSide> {
        h.iter().map(|s| s.unwrap_or(HeldSide::G)).collect()
    };

    if biactive.len() <= MAX_ENUMERATED_BIACTIVE {
        out.enumerated = true;
        let mut best: Option<(Number, Vec<HeldSide>)> = None;
        for mask in 0..(1usize << biactive.len()) {
            let held = held_of(&|k| {
                let bit = biactive.iter().position(|&b| b == k).unwrap();
                Some(if mask >> bit & 1 == 0 {
                    HeldSide::G
                } else {
                    HeldSide::H
                })
            });
            let Some((v, _)) = branch_lp(inp, &act, &held) else {
                out.verdict = BVerdict::NotChecked("a branch LP did not solve");
                return out;
            };
            out.branches_solved += 1;
            if best.as_ref().is_none_or(|(b, _)| v < *b) {
                best = Some((v, full(&held)));
            }
        }
        out.verdict = match best {
            Some((v, held)) if v < -band => BVerdict::NotB { held, decrease: v },
            _ => BVerdict::B,
        };
        return out;
    }

    // Too many to enumerate: the relaxed cone's direction names a branch.
    let relaxed = held_of(&|_| None);
    let Some((_, d)) = branch_lp(inp, &act, &relaxed) else {
        out.verdict = BVerdict::NotChecked("the relaxed LP did not solve");
        return out;
    };
    out.branches_solved += 1;
    let dg = |k: usize| act.g_side[k].0 * d[inp.pairs[k].var];
    let dh = |k: usize| {
        let i = inp.pairs[k].row;
        act.h_side[k].0
            * inp
                .jac
                .iter()
                .filter(|&&(r, _, _)| r == i)
                .map(|&(_, c, v)| v * d[c])
                .sum::<Number>()
    };
    let held = held_of(&|k| {
        Some(if dg(k) <= dh(k) {
            HeldSide::G
        } else {
            HeldSide::H
        })
    });
    let Some((v, _)) = branch_lp(inp, &act, &held) else {
        out.verdict = BVerdict::NotChecked("a branch LP did not solve");
        return out;
    };
    out.branches_solved += 1;
    out.verdict = if v < -band {
        BVerdict::NotB {
            held: full(&held),
            decrease: v,
        }
    } else {
        BVerdict::NotChecked(
            "too many biactive pairs to enumerate; the heuristic branch does not descend",
        )
    };
    out
}

/// One branch LP (`held[k] = None` on a biactive pair: both sides only
/// non-negative, the relaxed cone). Returns the optimum and the direction,
/// or `None` if the LP did not solve.
fn branch_lp(
    inp: &CertificateInput<'_>,
    act: &PairActivity,
    held: &[Option<HeldSide>],
) -> Option<(Number, Vec<Number>)> {
    use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
    let n = inp.x.len();
    let m = inp.g.len();
    let row_tol = inp.solver_tol.max(0.0).sqrt();
    let mut rows_of: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    for &(r, c, v) in inp.jac {
        rows_of[r].push((c, v));
    }
    let (mut lb, mut ub) = (vec![-1.0; n], vec![1.0; n]);
    let (mut a, mut b, mut g, mut h) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let eq = |coef: &[(usize, Number)], s: Number, a: &mut Vec<Triplet>, b: &mut Vec<Number>| {
        let r = b.len();
        a.extend(coef.iter().map(|&(c, v)| Triplet {
            row: r,
            col: c,
            val: s * v,
        }));
        b.push(0.0);
    };
    // `s·(coef·d) >= 0`, written `-s·coef·d <= 0`.
    let geq = |coef: &[(usize, Number)], s: Number, g: &mut Vec<Triplet>, h: &mut Vec<Number>| {
        let r = h.len();
        g.extend(coef.iter().map(|&(c, v)| Triplet {
            row: r,
            col: c,
            val: -s * v,
        }));
        h.push(0.0);
    };
    let mut pair_row = vec![None; m];
    let mut pair_var = vec![None; n];
    for (k, p) in inp.pairs.iter().enumerate() {
        pair_row[p.row] = Some(k);
        pair_var[p.var] = Some(k);
    }
    // Ordinary rows: the active ones, linearized.
    for i in (0..m).filter(|&i| pair_row[i].is_none()) {
        let (lo, hi) = (inp.g_l[i], inp.g_u[i]);
        let at_lo = lo > -INF && inp.g[i] - lo <= row_tol;
        let at_hi = hi < INF && hi - inp.g[i] <= row_tol;
        match (at_lo, at_hi) {
            (true, true) => eq(&rows_of[i], 1.0, &mut a, &mut b),
            (true, false) => geq(&rows_of[i], 1.0, &mut g, &mut h),
            (false, true) => geq(&rows_of[i], -1.0, &mut g, &mut h),
            (false, false) => {}
        }
    }
    // Ordinary bounds, and a pair variable's ordinary second bound.
    for j in 0..n {
        let (lo, hi) = (inp.x_l[j], inp.x_u[j]);
        let (mut at_lo, mut at_hi) = (
            lo > -INF && inp.x[j] - lo <= row_tol,
            hi < INF && hi - inp.x[j] <= row_tol,
        );
        if let Some(k) = pair_var[j] {
            if act.mixed[k] {
                continue; // both bounds are the pair's
            }
            // The pair's own bound is handled with the pair.
            if act.g_side[k].0 > 0.0 {
                at_lo = false;
            } else {
                at_hi = false;
            }
        }
        if at_lo {
            lb[j] = 0.0;
        }
        if at_hi {
            ub[j] = 0.0;
        }
    }
    // Pairs.
    for (k, p) in inp.pairs.iter().enumerate() {
        let (sg, sh) = (act.g_side[k].0, act.h_side[k].0);
        let gcoef = [(p.var, 1.0)];
        let hcoef = &rows_of[p.row];
        match held[k] {
            Some(HeldSide::G) => {
                lb[p.var] = 0.0;
                ub[p.var] = 0.0;
                if act.h_act[k] {
                    geq(hcoef, sh, &mut g, &mut h);
                }
            }
            Some(HeldSide::H) => {
                eq(hcoef, 1.0, &mut a, &mut b);
                if act.g_act[k] {
                    geq(&gcoef, sg, &mut g, &mut h);
                }
            }
            None => {
                geq(&gcoef, sg, &mut g, &mut h);
                geq(hcoef, sh, &mut g, &mut h);
            }
        }
    }
    let prob = QpProblem {
        n,
        p_lower: Vec::new(),
        c: inp.grad_f.to_vec(),
        a,
        b,
        g,
        h,
        lb,
        ub,
    };
    let opts = QpOptions::default();
    let sol = solve_qp_ipm(&prob, &opts, || {
        Box::new(pounce_feral::FeralSolverInterface::new())
            as Box<dyn pounce_linsol::SparseSymLinearSolverInterface>
    });
    matches!(sol.status, QpStatus::Optimal).then_some((sol.obj, sol.x))
}

/// The branch NLP's bounds for `held` (one per pair): the held side pinned
/// at zero, and, on a mixed pair held at a bound, the free row given the
/// sign the pair requires there. For [`pounce_nlp::mpcc::MpccTnlp::set_branch`].
pub fn branch_bounds(
    inp: &CertificateInput<'_>,
    held: &[HeldSide],
) -> pounce_nlp::mpcc::BranchBounds {
    let corner = inp.solver_tol.max(0.0).sqrt();
    let act = pair_activity(inp, corner);
    let mut out = pounce_nlp::mpcc::BranchBounds::default();
    for (k, p) in inp.pairs.iter().enumerate() {
        let ((sg, bnd), (_, c)) = (act.g_side[k], act.h_side[k]);
        match held[k] {
            HeldSide::G => {
                out.vars.push((p.var, bnd, bnd));
                if act.mixed[k] {
                    // H = s_G·g >= 0 at the held bound.
                    out.rows.push(if sg > 0.0 {
                        (p.row, 0.0, 2.0 * INF)
                    } else {
                        (p.row, -2.0 * INF, 0.0)
                    });
                }
            }
            HeldSide::H => out.rows.push((p.row, c, c)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Near-dependent columns, one of them pointing away from `b`: the
    /// least-squares solve on the passive set gives a non-positive
    /// coefficient, which must not be picked again forever.
    #[test]
    fn nnls_on_near_dependent_columns_terminates_at_the_fit() {
        let a = vec![1.0, 2.0, -1.0];
        let a2: Vec<Number> = a.iter().map(|v| v * (1.0 + 1e-13)).collect();
        let neg: Vec<Number> = a.iter().map(|v| -v).collect();
        let b = vec![2.0, 4.0, -2.0];
        let x = nnls(&[a.clone(), a2.clone(), neg], &b, 3);
        assert!(x.iter().all(|&v| v >= 0.0));
        let r: Number = (0..3)
            .map(|i| (b[i] - x[0] * a[i] - x[1] * a2[i] + x[2] * a[i]).powi(2))
            .sum();
        assert!(r.sqrt() < 1e-9, "residual {}", r.sqrt());
    }

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

    /// Certifies on both paths and holds them to the same class, so every
    /// point a test below checks is also a dense-vs-sparse parity check.
    fn run(m: &Model, x: &[Number], grad_f: &[Number], g: &[Number]) -> Certificate {
        let inp = CertificateInput {
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
            feas_tol: 1e-4,
        };
        let dense = certify_routed(&inp, Some(false));
        let sparse = certify_routed(&inp, Some(true));
        assert_eq!(
            (dense.class, dense.multipliers_unique),
            (sparse.class, sparse.multipliers_unique),
            "dense {dense:?}\nsparse {sparse:?}"
        );
        // The residuals agree far inside the tolerance the class is decided
        // against (the SVD reaches ~1e-16, the LDLᵀ solve ~1e-13).
        if dense.multipliers_unique && dense.residual.is_finite() {
            assert!(
                (dense.residual - sparse.residual).abs() <= 1e-6 * dense.resid_tol,
                "dense {dense:?}\nsparse {sparse:?}"
            );
        }
        dense
    }

    /// A mixed pair, `0 <= x <= 1 ⟂ g = y - x` with `y` and the row free,
    /// certified at each place it can be: at either bound with the sign the
    /// pair requires, strictly between with `g = 0`, at a bound with the wrong
    /// sign (infeasible: no row bound states it), and strictly between with
    /// `g != 0` (not complementary).
    #[test]
    fn a_mixed_pair_is_certified_from_the_bound_it_is_at() {
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![1.0, PI],
            g_l: vec![NI],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 0, -1.0), (0, 1, 1.0)],
        };
        let g = |x: &[Number]| vec![x[1] - x[0]];

        // x at 0, g = 2 >= 0: min (x - 0.5)² + (y - 2)², grad (-1, 0) held by
        // the bound alone.
        let x = [0.0, 2.0];
        let c = run(&m, &x, &[-1.0, 0.0], &g(&x));
        assert_eq!(c.class, StationarityClass::S, "{c:?}");
        assert!(c.feasible, "{c:?}");

        // x at 1, g = -1 <= 0: min (x - 2)² + y², grad (-2, 0).
        let x = [1.0, 0.0];
        let c = run(&m, &x, &[-2.0, 0.0], &g(&x));
        assert_eq!(c.class, StationarityClass::S, "{c:?}");

        // Strictly between, g = 0: the row is an equality there, grad = λ ∇g.
        let x = [0.5, 0.5];
        let c = run(&m, &x, &[-1.0, 1.0], &g(&x));
        assert_eq!(c.class, StationarityClass::S, "{c:?}");

        // x at 0 with g = -1 < 0: violates the pair; no row bound says so.
        let x = [0.0, -1.0];
        let c = run(&m, &x, &[0.0, 0.0], &g(&x));
        assert_eq!(c.class, StationarityClass::None, "{c:?}");
        assert!(
            !c.feasible && (c.primal_violation - 1.0).abs() < 1e-12,
            "{c:?}"
        );

        // Strictly between with g = 0.5: neither side at zero.
        let x = [0.5, 1.0];
        let c = run(&m, &x, &[0.0, 0.0], &g(&x));
        assert_eq!(
            (c.class, c.n_neither),
            (StationarityClass::None, 1),
            "{c:?}"
        );
    }

    /// A one-sided pair whose variable also has an ordinary second bound:
    /// `x >= 0 ⟂ g = y >= 0` with `x <= 1` a plain limit. `min (x - 2)² +
    /// (y - 1)²` has its optimum at `(1, 0)` -- `x` on the ordinary bound, not
    /// the pair's -- which is stationary only with that bound's multiplier.
    #[test]
    fn a_pair_variables_ordinary_second_bound_carries_a_multiplier() {
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![1.0, PI],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0)],
        };
        let x = [1.0, 0.0];
        let c = run(&m, &x, &[-2.0, -2.0], &[x[1]]);
        assert_eq!(c.class, StationarityClass::S, "{c:?}");
        // The pair's own bound is the lower one: at x = 0 with y = 1 the
        // gradient (−4, 0) is held by it.
        let x = [0.0, 1.0];
        let c = run(&m, &x, &[-4.0, 0.0], &[x[1]]);
        assert_eq!(c.class, StationarityClass::S, "{c:?}");
    }

    fn input<'a>(
        m: &'a Model,
        x: &'a [Number],
        grad_f: &'a [Number],
        g: &'a [Number],
    ) -> CertificateInput<'a> {
        CertificateInput {
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
            feas_tol: 1e-4,
        }
    }

    /// ralph1 at the origin, `min 2x - y`, `x >= 0`, `0 <= y ⟂ y - x >= 0`:
    /// weakly but not strongly stationary, and B-stationary -- both branch
    /// LPs have optimum 0. The check must not report a descent there.
    #[test]
    fn ralph1s_origin_is_b_stationary() {
        let m = Model {
            x_l: vec![0.0, 0.0],
            x_u: vec![PI, PI],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 1 }],
            jac: vec![(0, 0, -1.0), (0, 1, 1.0)],
        };
        let (x, gf, g) = ([0.0, 0.0], [2.0, -1.0], [0.0]);
        assert_eq!(run(&m, &x, &gf, &g).class, StationarityClass::W);
        let b = check_b_stationarity(&input(&m, &x, &gf, &g));
        assert_eq!(b.verdict, BVerdict::B, "{b:?}");
        assert_eq!(
            (b.n_biactive, b.branches_solved, b.enumerated),
            (1, 2, true)
        );
    }

    /// `min (x - 1)² + (y - 1)²`, `0 <= x ⟂ y >= 0` at the origin: biactive,
    /// and either branch descends (`∇f·d = -2` with `d` in the unit box).
    /// The branch bounds pin the held side.
    #[test]
    fn a_biactive_corner_with_descent_is_not_b_stationary() {
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![PI, PI],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0)],
        };
        let (x, gf, g) = ([0.0, 0.0], [-2.0, -2.0], [0.0]);
        let inp = input(&m, &x, &gf, &g);
        let b = check_b_stationarity(&inp);
        let BVerdict::NotB { held, decrease } = &b.verdict else {
            panic!("{b:?}")
        };
        assert!((decrease + 2.0).abs() < 1e-6, "{b:?}");
        let bb = branch_bounds(&inp, held);
        match held[0] {
            HeldSide::G => assert_eq!(bb.vars, vec![(0, 0.0, 0.0)], "{bb:?}"),
            HeldSide::H => assert_eq!(bb.rows, vec![(0, 0.0, 0.0)], "{bb:?}"),
        }
        // Away from the corner, a point on one branch whose gradient is held
        // by the pinned side is B-stationary: x = 0, y = 1 with ∇f = (-2, 0).
        let (x, gf, g) = ([0.0, 1.0], [-2.0, 0.0], [1.0]);
        assert_eq!(
            check_b_stationarity(&input(&m, &x, &gf, &g)).verdict,
            BVerdict::B
        );
    }

    /// Past [`MAX_ENUMERATED_BIACTIVE`], the relaxed LP's direction picks one
    /// branch: it finds the descent here, and could never certify B.
    #[test]
    fn past_the_enumeration_cap_the_heuristic_finds_a_descent() {
        let k = MAX_ENUMERATED_BIACTIVE + 1;
        let n = 2 * k;
        let m = Model {
            x_l: (0..n).map(|j| if j < k { 0.0 } else { NI }).collect(),
            x_u: vec![PI; n],
            g_l: vec![0.0; k],
            g_u: vec![PI; k],
            pairs: (0..k)
                .map(|i| ComplementarityPair { row: i, var: i })
                .collect(),
            jac: (0..k).map(|i| (i, k + i, 1.0)).collect(),
        };
        let (x, gf, g) = (vec![0.0; n], vec![-1.0; n], vec![0.0; k]);
        let b = check_b_stationarity(&input(&m, &x, &gf, &g));
        assert!(!b.enumerated && b.branches_solved == 2, "{b:?}");
        assert!(matches!(b.verdict, BVerdict::NotB { .. }), "{b:?}");
    }

    #[test]
    fn a_point_where_a_pair_is_not_complementary_is_not_checked() {
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![PI, PI],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0)],
        };
        let (x, gf, g) = ([1.0, 1.0], [0.0, 0.0], [1.0]);
        let b = check_b_stationarity(&input(&m, &x, &gf, &g));
        assert!(matches!(b.verdict, BVerdict::NotChecked(_)), "{b:?}");
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

    /// Past [`DENSE_ENTRIES`] a system with unique multipliers is certified
    /// sparsely, at any size; it used to be skipped past
    /// [`MAX_DENSE_ENTRIES`] (PR review: the stall guard was off for
    /// mid-size models). 2001 active bounds give a 2001 × 2001 system,
    /// 4 004 001 entries, past both limits. A wrong-signed bound multiplier
    /// is still caught there.
    #[test]
    fn a_large_system_with_unique_multipliers_is_certified_sparsely() {
        let n = 2001;
        let m = Model {
            x_l: vec![0.0; n],
            x_u: vec![PI; n],
            g_l: vec![],
            g_u: vec![],
            pairs: vec![],
            jac: vec![],
        };
        assert!(n * n > MAX_DENSE_ENTRIES);
        let mut grad = vec![1.0; n];
        let cert = |grad: &[Number]| {
            certify(&CertificateInput {
                x: &vec![0.0; n],
                grad_f: grad,
                g: &[],
                jac: &m.jac,
                x_l: &m.x_l,
                x_u: &m.x_u,
                g_l: &m.g_l,
                g_u: &m.g_u,
                pairs: &m.pairs,
                solver_tol: 1e-8,
                feas_tol: 1e-4,
            })
        };
        let c = cert(&grad);
        assert_eq!(
            (c.class, c.multipliers_unique),
            (StationarityClass::S, true),
            "{c:?}"
        );
        grad[7] = -1.0;
        let c = cert(&grad);
        assert_eq!(c.class, StationarityClass::None, "{c:?}");
    }

    /// A badly scaled column (PR review): columns `e₀` and `(1, ε)`, swept
    /// from well conditioned to exactly dependent. The sparse path's
    /// multipliers are exact (after refinement) at every ε > 0, and it calls
    /// the system rank-deficient only at ε = 0, where the factorization's
    /// inertia is wrong. The dense SVD draws the same line (its threshold is
    /// `ε_mach · max(n, k) · σ_max`), so class and uniqueness agree on both
    /// paths at every scale (`run` asserts it).
    #[test]
    fn a_badly_scaled_column_gets_the_same_verdict_on_both_paths() {
        for eps in [1e-3, 1e-5, 1e-7, 1e-9, 1e-12, 1e-14, 0.0] {
            let m = Model {
                x_l: vec![0.0, -2e19],
                x_u: vec![2e19, 2e19],
                g_l: vec![0.0],
                g_u: vec![2e19],
                pairs: vec![],
                jac: vec![(0, 0, 1.0), (0, 1, eps)],
            };
            // grad f = A (1, 1): multipliers +1 and +1, so S.
            let c = run(&m, &[0.0, 0.0], &[2.0, eps], &[0.0]);
            assert_eq!(c.class, StationarityClass::S, "{eps:e}: {c:?}");
            assert_eq!(c.multipliers_unique, eps != 0.0, "{eps:e}: {c:?}");
            let cols = vec![vec![(0, 1.0)], vec![(0, 1.0), (1, eps)]];
            match sparse_least_squares(2, &cols, &[2.0, eps]) {
                LeastSquares::Solved(theta) => {
                    assert!(eps != 0.0, "{theta:?}");
                    assert!(
                        theta.iter().all(|t| (t - 1.0).abs() < 1e-12),
                        "{eps:e}: {theta:?}"
                    );
                }
                other => assert!(
                    eps == 0.0 && other == LeastSquares::RankDeficient,
                    "{eps:e}: {other:?}"
                ),
            }
        }
    }

    /// One acceptance band on every path (PR review, MacMPEC): an IPM
    /// answer reproduces `∇f` only to about `sqrt(tol)` near the boundary,
    /// and the unique-multiplier path used to hold it to `max(1e-6, tol)`
    /// while the set search allowed the `sqrt(tol)` band too -- so `flp4-3`,
    /// `flp4-4` and `portfl-i-1`, optimal to f within 0.2%, read `none` with
    /// residuals of 1.7e-6 to 2.1e-5. Here a lone free variable with
    /// `∂f = r` leaves residual `r` (dual scale 1, band `1e-6 + 1e-4`).
    #[test]
    fn the_residual_band_is_the_same_on_the_unique_path() {
        let m = Model {
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![],
            g_u: vec![],
            pairs: vec![],
            jac: vec![],
        };
        // Inside the floor: S and tight. Inside the band only: S, not tight
        // (the driver keeps it and looks for a tighter answer). Outside: none.
        for (r, class, tight) in [
            (5e-7, StationarityClass::S, true),
            (2e-5, StationarityClass::S, false),
            (9e-5, StationarityClass::S, false),
            (2e-4, StationarityClass::None, false),
        ] {
            let c = run(&m, &[0.3], &[r], &[]);
            assert_eq!((c.class, c.tight), (class, tight), "{r:e}: {c:?}");
            assert!((c.resid_tol - (1e-6 + 1e-4)).abs() < 1e-15, "{c:?}");
        }
    }

    /// A point outside its own constraints is not certified (PR review,
    /// MacMPEC `pack-rig2*`: infeasibility verdicts whose points violate
    /// rows by 4.6e-4 to 9.9e-4 read "S"). Row `x₀ ≥ 1` at `x₀ = 1 − 5e-4`
    /// with `∇f = e₀`: the row's multiplier is +1, which would be S.
    #[test]
    fn a_point_violating_a_row_beyond_constr_viol_tol_is_not_certified() {
        let m = Model {
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![1.0],
            g_u: vec![2e19],
            pairs: vec![],
            jac: vec![(0, 0, 1.0)],
        };
        let x = 1.0 - 5e-4;
        let c = run(&m, &[x], &[1.0], &[x]);
        assert_eq!(
            (c.class, c.feasible),
            (StationarityClass::None, false),
            "{c:?}"
        );
        assert!((c.primal_violation - 5e-4).abs() < 1e-12, "{c:?}");
        // Inside constr_viol_tol (1e-4) the same point certifies.
        let x = 1.0 - 5e-5;
        let c = run(&m, &[x], &[1.0], &[x]);
        assert_eq!((c.class, c.feasible), (StationarityClass::S, true), "{c:?}");
        // A violated variable bound counts the same way.
        let m = Model {
            x_l: vec![0.0],
            g_l: vec![],
            g_u: vec![],
            jac: vec![],
            ..m
        };
        let c = run(&m, &[-2e-3], &[1.0], &[]);
        assert_eq!(
            (c.class, c.feasible),
            (StationarityClass::None, false),
            "{c:?}"
        );
    }

    /// A large system whose multipliers are a set is still not searched:
    /// the set search is dense. One active row `x₀ ≥ 0` duplicating the
    /// bound makes the 2001 bound columns plus the row dependent, and
    /// 2001 × 2002 entries is past [`MAX_DENSE_ENTRIES`].
    #[test]
    fn a_large_rank_deficient_system_is_not_searched() {
        let n = 2001;
        let c = certify(&CertificateInput {
            x: &vec![0.0; n],
            grad_f: &vec![1.0; n],
            g: &[0.0],
            jac: &[(0, 0, 1.0)],
            x_l: &vec![0.0; n],
            x_u: &vec![PI; n],
            g_l: &[0.0],
            g_u: &[2e19],
            pairs: &[],
            solver_tol: 1e-8,
            feas_tol: 1e-4,
        });
        assert_eq!(
            (c.class, c.multipliers_unique),
            (StationarityClass::NotSearched, false),
            "{c:?}"
        );
        assert!(n * (n + 1) > MAX_DENSE_ENTRIES);
    }

    /// The set search's acceptance band (PR review): with non-unique
    /// multipliers a point passes when the signed least-squares residual is
    /// within `resid_tol + sign_tol` — `1e-6 + 1e-4` here (`tol = 1e-8`,
    /// dual scale 1) — ~100× looser than `resid_tol` alone, by design (a
    /// sign inside the noise band is not evidence of descent). Two copies of
    /// the active row `x ≥ 0` make the multipliers a set; `∇f = −ε` leaves
    /// residual `ε` at the best non-negative multipliers. Pinned just inside
    /// and just outside the band so it cannot drift silently.
    #[test]
    fn the_set_search_band_is_resid_tol_plus_the_sign_band() {
        let m = Model {
            x_l: vec![NI],
            x_u: vec![PI],
            g_l: vec![0.0, 0.0],
            g_u: vec![PI, PI],
            pairs: vec![],
            jac: vec![(0, 0, 1.0), (1, 0, 1.0)],
        };
        let inside = run(&m, &[0.0], &[-5e-5], &[0.0, 0.0]);
        assert!(!inside.multipliers_unique, "{inside:?}");
        assert_eq!(inside.class, StationarityClass::S, "{inside:?}");
        let outside = run(&m, &[0.0], &[-2e-4], &[0.0, 0.0]);
        assert!(!outside.multipliers_unique, "{outside:?}");
        assert_eq!(outside.class, StationarityClass::None, "{outside:?}");
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
