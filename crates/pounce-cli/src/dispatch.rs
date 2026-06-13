//! Solver routing for the LP/QP/QCQP dispatch.
//!
//! See `dev-notes/lp-qp-routing.md`. This module sits between problem
//! loading and the call to `optimize_tnlp`. It does three things:
//!
//! 1. **Classify** the parsed problem into a [`ProblemClass`] by walking
//!    the nonlinear expression trees the `.nl` reader already produced.
//! 2. **Resolve** that class against the user's `solver_selection`
//!    option into a [`SolverChoice`].
//! 3. **Dispatch** to the chosen solver (in `main.rs`).
//!
//! All solvers are wired: `auto` routes an LP/convex-QP to `pounce-convex`'s
//! interior-point solver, a convex QCQP to the same crate's conic (SOCP)
//! driver, and everything else to the existing filter-IPM (`Nlp`).
//!
//! ## Classification
//!
//! The `.nl` format has no dedicated quadratic section: each row's
//! linear part lives in the `G`/`J` coefficient segments (already split
//! out into [`NlProblem::obj_linear`] / [`NlProblem::con_linear`]),
//! while any higher-order term — including a QP's quadratic terms — is
//! written into the nonlinear expression tree as `Mul`/`Pow` nodes. So:
//!
//! - no nonlinear parts at all → **LP**;
//! - all nonlinear parts are degree-2 polynomials → **QP** family
//!   (convex / nonconvex / QCQP split by curvature);
//! - anything else (transcendental, higher degree) → **NLP**.
//!
//! ### Conservative fallback (correctness guard)
//!
//! Misclassifying an indefinite or non-quadratic problem *into* a convex
//! solver would return a spurious KKT point as if globally optimal.
//! Whenever the walk cannot *prove* the stronger class, the classifier
//! falls back to the more general one, ultimately `Nlp`. The convexity
//! (PSD) test uses a tolerance and routes "inconclusive within
//! tolerance" to the safe side, never to the convex path.

use crate::nl_reader::{BinOp, Expr, NlProblem, UnaryOp};
use std::collections::BTreeMap;

/// Tolerance for the smallest-eigenvalue sign test in the convexity
/// check. A Hessian eigenvalue below `-PSD_TOL` is treated as a genuine
/// negative direction (nonconvex); within `±PSD_TOL` it is treated as
/// zero. Scaled tolerances would be better once we have problem scaling
/// in this path; a fixed absolute tolerance is adequate here and errs
/// toward the safe (more general) class.
const PSD_TOL: f64 = 1e-9;

/// Size budget (`n · m`) above which a convex QCQP is routed to the general
/// NLP solver instead of the conic (SOCP) interior-point path.
///
/// The QCQP→SOCP reformulation ([`crate::qp_extract::extract_socp_with_map`])
/// and the conic solve both scale with the problem's variable × constraint
/// product; for the very large convex QCQPs in the mittelmann set
/// (`nql180` ≈ 1.3e5 vars × 1.3e5 cons, `qssp180` ≈ 2.0e5 × 1.3e5) the
/// reformulation alone burns the entire CPU budget before the solver starts.
/// The pre-classifier baseline routed these to the NLP filter-IPM, which
/// solves them in well under the time limit (`qssp180` 27 iters, `nql180`
/// 44 iters). Above this budget we do the same: a convex QCQP is still a
/// valid NLP, so the fallback is sound — it only forgoes the conic
/// specialization on a scale the conic path is not yet tuned for.
///
/// `1e8` keeps the conic path for small-to-moderate QCQPs (e.g. 1e4 × 1e4)
/// while bounding the reformulation cost to roughly a second.
const SOCP_SIZE_BUDGET: u64 = 100_000_000;

/// Per-constraint coupling budget for the QCQP→SOCP conic path.
///
/// The `n · m` [`SOCP_SIZE_BUDGET`] catches QCQPs that are large in the
/// *problem* dimensions, but a problem can have a small `n · m` and still be
/// ruinously expensive to put in conic form: each convex quadratic *row*
/// `½xᵀQx ≤ b` is reformulated to a second-order cone via a factorization of
/// its Hessian `Q` ([`crate::qp_extract::extract_socp_with_map`]), which costs
/// `O(k³)` in the number of variables `k` that couple inside that one
/// constraint. The mittelmann `qcqp1000-*` rows have only a handful of
/// constraints (tiny `n · m`) but each couples ~1000 variables, so the
/// per-row factorization alone exhausts the CPU budget before the conic solve
/// starts.
///
/// When any single quadratic constraint couples more than this many active
/// variables we route the whole QCQP to the general NLP filter-IPM, which
/// solves it soundly without the conic reformulation — exactly what the
/// classifier did for these rows before the convexity certificate was made
/// cheap. A *diagonal* (separable) constraint Hessian is exempt: it is
/// SOC-representable in `O(nnz)` with no factorization, so its size is
/// harmless. This guard governs only the conic *reformulation* cost; the
/// convexity test itself is the cheap sparse factorization in
/// [`coupled_hessian_is_psd`].
const QCQP_SOCP_COUPLED_VARS: usize = 256;

/// The `.nl` "infinity" sentinel for a missing bound: AMPL writes ±1e20-ish
/// and upstream Ipopt treats any magnitude ≥ 1e19 as infinite. Used to read
/// a quadratic constraint's *sense* (one-sided `≤` vs. equality / range / `≥`)
/// when deciding whether a QCQP is convex — see [`classify_problem`].
const NL_INF: f64 = 1e19;

#[inline]
fn is_finite_bound(v: f64) -> bool {
    v.abs() < NL_INF
}

/// The mathematical class of a loaded problem, from most to least
/// specialized. See the module docs and `dev-notes/lp-qp-routing.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// Linear objective, linear constraints.
    Lp,
    /// Convex quadratic objective, linear constraints (Hessian PSD).
    ConvexQp,
    /// Convex quadratic objective and/or convex quadratic constraints.
    /// SOCP-representable; routes to the conic (SOCP) interior-point solver.
    ConvexQcqp,
    /// Quadratic but with an indefinite Hessian somewhere. Falls through
    /// to the NLP solver for a local minimum.
    NonconvexQp,
    /// General nonlinear (transcendental terms, higher-degree
    /// polynomials, or anything the classifier cannot prove quadratic).
    Nlp,
}

impl ProblemClass {
    /// Human-readable name for diagnostics and the
    /// forced-solver-mismatch error message.
    pub fn name(self) -> &'static str {
        match self {
            ProblemClass::Lp => "LP",
            ProblemClass::ConvexQp => "convex QP",
            ProblemClass::ConvexQcqp => "convex QCQP",
            ProblemClass::NonconvexQp => "nonconvex QP",
            ProblemClass::Nlp => "NLP",
        }
    }
}

/// The resolved solver to dispatch to, after combining a
/// [`ProblemClass`] with the `solver_selection` option.
///
/// `auto` resolves an LP/convex-QP to [`SolverChoice::LpIpm`]/[`SolverChoice::QpIpm`],
/// a convex QCQP to [`SolverChoice::SocpIpm`], and everything else to
/// [`SolverChoice::Nlp`]; a forced `solver_selection` can pin any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverChoice {
    /// The existing Wächter-Biegler filter-IPM.
    Nlp,
    /// LP interior-point in `pounce-convex`.
    LpIpm,
    /// Convex-QP interior-point in `pounce-convex`.
    QpIpm,
    /// Conic (SOCP) IPM in `pounce-convex`: convex QCQP, reformulated to
    /// second-order cones.
    SocpIpm,
    /// Active-set QP in `pounce-qp` (parallel track).
    QpActiveSet,
}

impl SolverChoice {
    /// Human-readable description of the dispatched solver, for the
    /// banner-level "Solving as …" log line. Names the algorithm and the
    /// crate that implements it so a reader can tell which of pounce's
    /// solvers actually ran.
    pub fn describe(self) -> &'static str {
        match self {
            SolverChoice::Nlp => "NLP filter line-search interior-point (pounce-nlp)",
            SolverChoice::LpIpm => "LP interior-point (pounce-convex)",
            SolverChoice::QpIpm => "convex QP interior-point (pounce-convex)",
            SolverChoice::SocpIpm => "convex QCQP conic interior-point (pounce-convex)",
            SolverChoice::QpActiveSet => "active-set QP (pounce-qp)",
        }
    }
}

/// Parsed `solver_selection` option value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverSelection {
    /// Pick the most specialized solver matching the class. Default.
    Auto,
    /// Force the NLP solver regardless of class (current behavior).
    Nlp,
    /// Force IPM-LP; error if the problem is not an LP.
    LpIpm,
    /// Force IPM-QP; error if the problem is not LP/convex-QP.
    QpIpm,
    /// Force the conic (SOCP) IPM; error if the problem is not a convex
    /// LP / QP / QCQP (all of which the conic solver handles).
    Socp,
    /// Force active-set QP; error if the problem is not LP/convex-QP.
    QpActiveSet,
}

impl SolverSelection {
    /// Parse the `solver_selection` option string. Returns `None` for an
    /// unrecognized value so the caller can surface a tidy error.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(SolverSelection::Auto),
            "nlp" => Some(SolverSelection::Nlp),
            "lp-ipm" => Some(SolverSelection::LpIpm),
            "qp-ipm" => Some(SolverSelection::QpIpm),
            "socp" => Some(SolverSelection::Socp),
            "qp-active-set" => Some(SolverSelection::QpActiveSet),
            _ => None,
        }
    }

    /// The accepted values, for error messages and option registration.
    pub const VALUES: &'static [&'static str] =
        &["auto", "nlp", "lp-ipm", "qp-ipm", "socp", "qp-active-set"];
}

/// Classify a parsed `.nl` problem.
///
/// Works off the already-split linear / nonlinear representation in
/// [`NlProblem`]: a row contributes to the class only through its
/// nonlinear `Expr` (the linear part is, by construction, linear). The
/// classifier is deliberately conservative — see the module docs.
pub fn classify_problem(prob: &NlProblem) -> ProblemClass {
    // Fast path: no nonlinear parts anywhere ⇒ LP. (Header-equivalent:
    // n_nl_objs == 0 && n_nl_cons == 0.)
    let obj_nl = !is_trivially_zero(&prob.obj_nonlinear);
    let cons_nl = prob.con_nonlinear.iter().any(|e| !is_trivially_zero(e));
    if !obj_nl && !cons_nl {
        return ProblemClass::Lp;
    }

    // Objective curvature.
    let obj_quad = match analyze_quadratic(&prob.obj_nonlinear, prob.n) {
        Some(q) => q,
        // Objective has a non-quadratic nonlinear term ⇒ NLP.
        None => return ProblemClass::Nlp,
    };

    // Constraint curvature. A quadratic constraint makes this a QCQP;
    // any non-quadratic constraint term makes the whole problem NLP.
    let mut any_quadratic_constraint = false;
    for c in &prob.con_nonlinear {
        if is_trivially_zero(c) {
            continue;
        }
        match analyze_quadratic(c, prob.n) {
            Some(q) if q.is_empty() => {} // purely linear after all
            Some(_) => any_quadratic_constraint = true,
            None => return ProblemClass::Nlp,
        }
    }

    // Objective Hessian definiteness, as the *minimizer* sees it. A
    // `maximize` problem is internally negated to a minimization, so a
    // concave-up (PSD-Hessian) maximize is a nonconvex minimize. Test the
    // sense-adjusted Hessian, not the raw one, or maximize-of-convex slips
    // through to the convex IPM and produces a wrong (max/saddle) answer.
    if !obj_quad.is_empty() {
        let effective: QuadHessian = if prob.minimize {
            obj_quad.clone()
        } else {
            obj_quad.iter().map(|(k, v)| (*k, -v)).collect()
        };
        if !hessian_is_psd(&effective, prob.n) {
            return ProblemClass::NonconvexQp;
        }
    }

    if any_quadratic_constraint {
        // Convex QCQP requires every quadratic constraint to be convex *as a
        // feasible set*, not merely to have a PSD Hessian. A quadratic
        // `g(x) = ½xᵀQx + … ` carves a convex region only when it is a
        // one-sided **upper** bound `g(x) ≤ g_u` *and* `Q ⪰ 0`. The other
        // senses are nonconvex even with a PSD Hessian:
        //   - `g(x) ≥ g_l` (finite lower bound): the super-level set of a
        //     convex function is nonconvex;
        //   - a quadratic equality `g(x) = c`;
        //   - a two-sided range `g_l ≤ g(x) ≤ g_u` (includes the `≥` side).
        // This sense test matters now that ConvexQcqp is dispatched to the
        // conic solver (it is SOC-representable only in the convex case); a
        // misclassified nonconvex row would return a spurious "optimum".
        // Anything not provably convex falls back to NLP (sound: the
        // filter-IPM finds a local minimum either way).
        for (row, c) in prob.con_nonlinear.iter().enumerate() {
            if is_trivially_zero(c) {
                continue;
            }
            match analyze_quadratic(c, prob.n) {
                Some(q) if q.is_empty() => {} // purely linear after all
                Some(q) => {
                    let lo = prob.g_l[row];
                    let hi = prob.g_u[row];
                    let vacuous = !is_finite_bound(lo) && !is_finite_bound(hi);
                    let upper_only = is_finite_bound(hi) && !is_finite_bound(lo);
                    if vacuous {
                        // Free row: imposes nothing, so it cannot make the
                        // problem nonconvex. Ignore it.
                        continue;
                    }
                    // Convexity (cheap sparse certificate) gates the QCQP
                    // class; the per-row coupling guard then gates the *conic*
                    // path: a convex but heavily-coupled constraint Hessian is
                    // ruinous to put in SOC form, so route the whole QCQP to
                    // NLP (which solves it soundly) rather than burn the budget
                    // in the reformulation — the mittelmann `qcqp1000-*` rows.
                    if !upper_only
                        || !hessian_is_psd(&q, prob.n)
                        || qcqp_constraint_too_costly_for_socp(&q)
                    {
                        return ProblemClass::Nlp;
                    }
                }
                None => return ProblemClass::Nlp,
            }
        }
        // A convex QCQP whose scale exceeds the conic path's budget falls
        // back to NLP: the QCQP→SOCP reformulation and conic solve scale with
        // `n · m`, and beyond this the setup alone exhausts the CPU budget
        // (the mittelmann `nql180`/`qssp180` regression). NLP solves the same
        // problem soundly — see `SOCP_SIZE_BUDGET`.
        if (prob.n as u64).saturating_mul(prob.m as u64) > SOCP_SIZE_BUDGET {
            return ProblemClass::Nlp;
        }
        return ProblemClass::ConvexQcqp;
    }

    // Quadratic (or linear) convex objective with linear constraints.
    if obj_quad.is_empty() {
        // Objective nonlinear part collapsed to nothing quadratic and no
        // constraints are quadratic — it was effectively linear.
        ProblemClass::Lp
    } else {
        ProblemClass::ConvexQp
    }
}

/// Resolve a [`ProblemClass`] and a [`SolverSelection`] into the solver
/// to dispatch to, or an error string when a forced selection does not
/// match the detected class.
///
/// `auto` routes LP / convex QP to the convex IPM (`QpIpm`) and convex
/// QCQP to the conic IPM (`SocpIpm`); nonconvex QP and general NLP resolve
/// to `Nlp`. A forced selection that does not match the detected class is
/// rejected with a clear message. (`QpActiveSet` is accepted for LP / convex
/// QP and dispatched to the active-set SQP engine — see `main.rs`.)
pub fn resolve_solver(
    class: ProblemClass,
    selection: SolverSelection,
) -> Result<SolverChoice, String> {
    use ProblemClass as P;
    use SolverSelection as S;

    // Is this class within the convex-QP family (LP or convex QP)?
    let is_lp = class == P::Lp;
    let is_convex_qp = matches!(class, P::Lp | P::ConvexQp);
    // The conic solver handles the whole convex cone family: LP, convex QP,
    // and (reformulated to second-order cones) convex QCQP.
    let is_conic = matches!(class, P::Lp | P::ConvexQp | P::ConvexQcqp);

    match selection {
        // `auto`: route LP and convex QP to the specialized convex IPM
        // (`pounce-convex`) and convex QCQP to the same crate's conic
        // (SOCP) IPM; nonconvex QP and general NLP fall through to the NLP
        // filter-IPM. LP is solved by the same QP IPM (P = 0), so it
        // resolves to `QpIpm` rather than a distinct LP entry point.
        S::Auto => match class {
            P::Lp | P::ConvexQp => Ok(SolverChoice::QpIpm),
            P::ConvexQcqp => Ok(SolverChoice::SocpIpm),
            _ => Ok(SolverChoice::Nlp),
        },
        S::Nlp => Ok(SolverChoice::Nlp),
        S::LpIpm => {
            if is_lp {
                Ok(SolverChoice::LpIpm)
            } else {
                Err(mismatch_msg(class, "lp-ipm", "an LP"))
            }
        }
        S::QpIpm => {
            if is_convex_qp {
                Ok(SolverChoice::QpIpm)
            } else {
                Err(mismatch_msg(class, "qp-ipm", "an LP or convex QP"))
            }
        }
        S::Socp => {
            if is_conic {
                Ok(SolverChoice::SocpIpm)
            } else {
                Err(mismatch_msg(class, "socp", "a convex LP, QP, or QCQP"))
            }
        }
        S::QpActiveSet => {
            if is_convex_qp {
                Ok(SolverChoice::QpActiveSet)
            } else {
                Err(mismatch_msg(class, "qp-active-set", "an LP or convex QP"))
            }
        }
    }
}

fn mismatch_msg(class: ProblemClass, forced: &str, expected: &str) -> String {
    format!(
        "problem class {} does not match forced solver {} (expected {})",
        class.name(),
        forced,
        expected
    )
}

// ---------------------------------------------------------------------
// Quadratic-form analysis
// ---------------------------------------------------------------------

/// The symmetric Hessian of a quadratic form, stored as a sparse upper-
/// triangular (i ≤ j) map of `(i, j) -> ∂²/∂xᵢ∂xⱼ`. Empty means the
/// expression is (at most) linear.
pub(crate) type QuadHessian = BTreeMap<(usize, usize), f64>;

/// Full quadratic read-out: `(Hessian, [(var, linear coef), …], constant)`.
/// The linear and constant parts are the pieces AMPL/Pyomo fold into the
/// nonlinear objective tree (see [`analyze_quadratic_full`]).
pub(crate) type QuadForm = (QuadHessian, Vec<(usize, f64)>, f64);

/// Attempt to read an expression as a polynomial of total degree ≤ 2 and
/// return its Hessian (constant, since the form is quadratic). Returns
/// `None` if the expression contains any term the classifier cannot
/// prove is degree-≤2 polynomial (transcendental ops, division by a
/// non-constant, `Pow` with exponent ∉ {0,1,2}, products of degree > 2,
/// external calls, …). `None` ⇒ treat as general nonlinear.
pub(crate) fn analyze_quadratic(e: &Expr, n: usize) -> Option<QuadHessian> {
    analyze_quadratic_full(e, n).map(|(h, _, _)| h)
}

/// Like [`analyze_quadratic`] but also returns the degree-1 (linear)
/// coefficients *and* the degree-0 (constant) term of the form:
/// `(Hessian, [(var, coef), …], constant)`.
///
/// AMPL folds the linear part of a nonlinear term into the objective's
/// nonlinear expression tree (the `−6·x₀` of `(x₀−3)²`, say) rather than
/// the linear section. Callers building the QP objective vector `c` must
/// add these in, exactly as the NLP path's `eval_f` sums the linear
/// section *and* the nonlinear tree — otherwise the linear shift is
/// silently dropped and the convex solve minimizes the wrong objective.
///
/// The **constant** is returned for the same reason: AMPL/Pyomo also fold
/// the objective's degree-0 term into the nonlinear tree (the `+9` of
/// `(x₀−3)²`), where it does *not* land in `NlProblem::obj_constant`. It
/// is irrelevant to the minimizer but is part of the *reported objective
/// value*; dropping it makes the convex solve report an objective off by
/// that constant versus the NLP path (see `qp_extract`).
pub(crate) fn analyze_quadratic_full(e: &Expr, _n: usize) -> Option<QuadForm> {
    let poly = to_poly(e)?;
    if poly.max_degree() > 2 {
        return None;
    }
    let mut h: QuadHessian = BTreeMap::new();
    let mut lin: Vec<(usize, f64)> = Vec::new();
    let mut constant = 0.0;
    for (vars, coef) in &poly.terms {
        match vars.as_slice() {
            // Constant term: no gradient/Hessian contribution, but it is
            // part of the objective *value* — accumulate, don't drop.
            [] => constant += *coef,
            // Linear term c·xᵢ.
            [i] => lin.push((*i, *coef)),
            // Quadratic term c·xᵢ·xⱼ.
            [i, j] => {
                let (i, j) = (*i.min(j), *i.max(j));
                // ∂²(c·xᵢxⱼ)/∂xᵢ∂xⱼ = c for i≠j; ∂²(c·xᵢ²)/∂xᵢ² = 2c.
                let contrib = if i == j { 2.0 * coef } else { *coef };
                *h.entry((i, j)).or_insert(0.0) += contrib;
            }
            _ => return None,
        }
    }
    // Drop explicit zeros so `is_empty()` means "linear".
    h.retain(|_, v| v.abs() > 0.0);
    Some((h, lin, constant))
}

/// A multivariate polynomial as a map from a sorted variable-index
/// multiset (the monomial) to its coefficient. `[]` is the constant
/// term, `[i]` is `xᵢ`, `[i, i]` is `xᵢ²`, `[i, j]` is `xᵢxⱼ`.
#[derive(Debug, Clone, Default)]
struct Poly {
    terms: BTreeMap<Vec<usize>, f64>,
}

impl Poly {
    fn constant(c: f64) -> Self {
        let mut terms = BTreeMap::new();
        if c != 0.0 {
            terms.insert(Vec::new(), c);
        }
        Poly { terms }
    }

    fn var(i: usize) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(vec![i], 1.0);
        Poly { terms }
    }

    fn max_degree(&self) -> usize {
        self.terms.keys().map(|m| m.len()).max().unwrap_or(0)
    }

    fn as_constant(&self) -> Option<f64> {
        match self.terms.len() {
            0 => Some(0.0),
            1 => self.terms.get(&Vec::new()).copied(),
            _ => None,
        }
    }

    fn add(mut self, other: &Poly) -> Poly {
        for (m, c) in &other.terms {
            *self.terms.entry(m.clone()).or_insert(0.0) += c;
        }
        self.prune();
        self
    }

    fn neg(mut self) -> Poly {
        for c in self.terms.values_mut() {
            *c = -*c;
        }
        self
    }

    fn scale(mut self, s: f64) -> Poly {
        if s == 0.0 {
            return Poly::default();
        }
        for c in self.terms.values_mut() {
            *c *= s;
        }
        self
    }

    /// Multiply two polynomials, bailing (`None`) if any product
    /// monomial would exceed total degree 2 — past that the classifier
    /// gives up and the caller routes to NLP.
    fn mul(&self, other: &Poly) -> Option<Poly> {
        let mut out = Poly::default();
        for (ma, ca) in &self.terms {
            for (mb, cb) in &other.terms {
                if ma.len() + mb.len() > 2 {
                    return None;
                }
                let mut m = ma.clone();
                m.extend_from_slice(mb);
                m.sort_unstable();
                *out.terms.entry(m).or_insert(0.0) += ca * cb;
            }
        }
        out.prune();
        Some(out)
    }

    fn prune(&mut self) {
        self.terms.retain(|_, c| c.abs() > 0.0);
    }
}

/// Lower an `Expr` to a [`Poly`] of total degree ≤ 2, or `None` if it
/// contains anything outside that class. `Cse` nodes are inlined (they
/// are mathematically equivalent to their body).
fn to_poly(e: &Expr) -> Option<Poly> {
    match e {
        Expr::Const(c) => Some(Poly::constant(*c)),
        Expr::Var(i) => Some(Poly::var(*i)),
        Expr::Cse(body) => to_poly(body),
        Expr::Sum(items) => {
            // Accumulate every monomial into one map, pruning ONCE at the
            // end. The previous `acc = acc.add(&to_poly(it)?)` called the
            // self-pruning `add` per item, and `prune` rescans the entire
            // accumulated map, making an N-term sum O(N²). On QCQP forms
            // (a quadratic over n vars expands to up to ~n² monomials) this
            // hung the `solver_selection=auto` classifier for >300 s before
            // the solver ever started. Merge-then-prune is O(N log N).
            let mut acc = Poly::default();
            for it in items {
                let p = to_poly(it)?;
                for (m, c) in &p.terms {
                    *acc.terms.entry(m.clone()).or_insert(0.0) += c;
                }
            }
            acc.prune();
            Some(acc)
        }
        Expr::Unary(op, a) => match op {
            UnaryOp::Neg => Some(to_poly(a)?.neg()),
            // Everything else is transcendental / non-polynomial.
            _ => None,
        },
        Expr::Binary(op, a, b) => {
            let pa = to_poly(a)?;
            let pb = to_poly(b)?;
            match op {
                BinOp::Add => Some(pa.add(&pb)),
                BinOp::Sub => Some(pa.add(&pb.neg())),
                BinOp::Mul => pa.mul(&pb),
                BinOp::Div => {
                    // Division is polynomial only by a nonzero constant.
                    let d = pb.as_constant()?;
                    if d == 0.0 {
                        None
                    } else {
                        Some(pa.scale(1.0 / d))
                    }
                }
                BinOp::Pow => {
                    // Polynomial only for constant integer exponents in
                    // {0, 1, 2}.
                    let exp = pb.as_constant()?;
                    if exp == 0.0 {
                        Some(Poly::constant(1.0))
                    } else if exp == 1.0 {
                        Some(pa)
                    } else if exp == 2.0 {
                        pa.mul(&pa)
                    } else {
                        None
                    }
                }
                // atan2 and any other binary opcodes are non-polynomial.
                _ => None,
            }
        }
        // External function calls are opaque ⇒ not provably polynomial.
        Expr::Funcall { .. } => None,
        // Comparisons, logicals, conditionals, and n-ary min/max (the
        // smooth-/control-flow `.nl` opcodes) are non-polynomial ⇒ not a
        // convex QP, so the classifier routes them to the NLP solver.
        _ => None,
    }
}

/// True if the expression is the literal constant zero the `.nl` reader
/// uses for "no nonlinear part".
fn is_trivially_zero(e: &Expr) -> bool {
    matches!(e, Expr::Const(c) if *c == 0.0)
}

// ---------------------------------------------------------------------
// PSD test
// ---------------------------------------------------------------------

/// Number of distinct variables that couple inside a quadratic form — the
/// dimension `k` of the matrix that would be factored.
fn hessian_active_vars(h: &QuadHessian) -> usize {
    let mut active: Vec<usize> = Vec::with_capacity(2 * h.len());
    for (i, j) in h.keys() {
        active.push(*i);
        active.push(*j);
    }
    active.sort_unstable();
    active.dedup();
    active.len()
}

/// True when reformulating this *convex* quadratic constraint to a
/// second-order cone would be too costly — a *coupled* (off-diagonal) Hessian
/// over more than [`QCQP_SOCP_COUPLED_VARS`] active variables, whose per-row
/// `O(k³)` factorization dominates the budget. A purely diagonal constraint
/// Hessian is exempt (SOC-representable in `O(nnz)`). Callers route such a
/// QCQP to the general NLP solver instead of the conic path. This is about
/// the *reformulation* cost, not convexity: the constraint is already known
/// convex (PSD) when this is consulted.
fn qcqp_constraint_too_costly_for_socp(h: &QuadHessian) -> bool {
    let has_offdiag = h.keys().any(|(i, j)| i != j);
    has_offdiag && hessian_active_vars(h) > QCQP_SOCP_COUPLED_VARS
}

/// Is the (symmetric, sparse) Hessian positive semidefinite?
///
/// A purely diagonal Hessian is settled in `O(nnz)` by sign — its
/// eigenvalues *are* its diagonal entries — with no factorization at all;
/// this keeps large separable / least-squares QPs cheap. A *coupled*
/// Hessian is certified by a sparse symmetric factorization (see
/// [`coupled_hessian_is_psd`]): feral's LDLᵀ reports the matrix inertia in
/// roughly `O(nnz · fill)`, so even the large but sparse coupled Hessians of
/// the CVXQP family (n ≈ 1000) are classified in well under the solve cost —
/// no dense `k×k` allocation and no `O(k³)` eigensolve. Returns `true` only
/// when the smallest eigenvalue is `≥ -PSD_TOL`; an indefinite or
/// inconclusive result returns `false`, routing to the safe (more general)
/// class.
fn hessian_is_psd(h: &QuadHessian, _n: usize) -> bool {
    if h.is_empty() {
        return true; // zero matrix is PSD (the linear case)
    }
    // Fast path: a diagonal Hessian is PSD iff every diagonal entry is
    // `≥ -PSD_TOL`. No factorization — essential for large but separable
    // objectives, where the answer is trivial.
    if h.keys().all(|(i, j)| i == j) {
        return h.values().all(|v| *v >= -PSD_TOL);
    }
    coupled_hessian_is_psd(h)
}

/// PSD certificate for a *coupled* Hessian via a sparse symmetric
/// factorization.
///
/// The test is positive-definiteness of the `ε`-shifted matrix `H + ε·I`
/// with `ε = PSD_TOL`. A genuinely-PSD `H` (smallest eigenvalue `λ_min ≥ 0`,
/// even a singular one) becomes strictly positive definite after the shift,
/// so feral factors it with no negative pivots (`inertia.negative == 0`); a
/// truly indefinite `H` with `λ_min < -PSD_TOL` keeps a strictly-negative
/// shifted eigenvalue and yields `negative > 0`. The `negative == 0` test on
/// the shifted matrix is therefore exactly `λ_min ≥ -PSD_TOL` — the same
/// tolerance the dense path used — and it scales to large sparse Hessians
/// because the factorization cost tracks the nonzero/fill count, not a dense
/// `k³`.
///
/// The Hessian is compressed to its active variable set so the factored
/// dimension is `k` (the number of distinct variables in the form). The
/// [`QuadHessian`] is upper-triangular (`i ≤ j`); feral wants the lower
/// triangle (`row ≥ col`), so each entry `(i, j)` is emitted at
/// `(row = j, col = i)`. Every active diagonal is seeded with `ε` (the shift;
/// `from_triplets` sums it with any diagonal entry already in `H`), which
/// also guarantees no structurally empty column. A non-`Success`
/// factorization (singular/fatal — should not occur given the strictly-PD
/// shift, but possible on a pathological form) is treated conservatively as
/// not-provably-PSD.
fn coupled_hessian_is_psd(h: &QuadHessian) -> bool {
    use feral::{CscMatrix, FactorStatus, Solver};

    // Compress to the active variable set so the factored dimension is `k`.
    let mut active: Vec<usize> = Vec::with_capacity(2 * h.len());
    for (i, j) in h.keys() {
        active.push(*i);
        active.push(*j);
    }
    active.sort_unstable();
    active.dedup();
    let k = active.len();
    let idx = |v: usize| active.binary_search(&v).unwrap();

    // Lower-triangle triplets: H's entry (i ≤ j) maps to (row = j, col = i).
    // Capacity covers H's nonzeros plus one ε-shift per active diagonal.
    let mut rows: Vec<usize> = Vec::with_capacity(h.len() + k);
    let mut cols: Vec<usize> = Vec::with_capacity(h.len() + k);
    let mut vals: Vec<f64> = Vec::with_capacity(h.len() + k);
    for ((i, j), v) in h {
        let (ri, rj) = (idx(*i), idx(*j));
        // i ≤ j by the upper-tri convention, so rj ≥ ri ⇒ lower triangle.
        rows.push(rj);
        cols.push(ri);
        vals.push(*v);
    }
    // εI shift: seed every active diagonal (summed with H's own diagonal).
    for d in 0..k {
        rows.push(d);
        cols.push(d);
        vals.push(PSD_TOL);
    }

    let mat = match CscMatrix::from_triplets(k, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(_) => return false, // malformed ⇒ be conservative
    };
    let mut solver = Solver::new();
    match solver.factor(&mat, None) {
        FactorStatus::Success => {
            // PD ⟺ no negative pivots in the LDLᵀ of the ε-shifted matrix.
            solver.inertia().map(|i| i.negative == 0).unwrap_or(false)
        }
        // Singular / wrong-inertia / fatal: cannot certify ⇒ safe fallback.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::parse_nl_text;

    // --- SolverSelection parsing ---

    #[test]
    fn parse_selection_values() {
        assert_eq!(SolverSelection::parse("auto"), Some(SolverSelection::Auto));
        assert_eq!(SolverSelection::parse("nlp"), Some(SolverSelection::Nlp));
        assert_eq!(
            SolverSelection::parse("lp-ipm"),
            Some(SolverSelection::LpIpm)
        );
        assert_eq!(
            SolverSelection::parse("qp-ipm"),
            Some(SolverSelection::QpIpm)
        );
        assert_eq!(
            SolverSelection::parse("qp-active-set"),
            Some(SolverSelection::QpActiveSet)
        );
        assert_eq!(SolverSelection::parse("lp-simplex"), None);
        assert_eq!(SolverSelection::parse("bogus"), None);
    }

    // --- resolve_solver: auto routes LP/convex-QP to the convex IPM,
    // everything else to NLP ---

    #[test]
    fn auto_routes_convex_qp_family_to_qp_ipm() {
        assert_eq!(
            resolve_solver(ProblemClass::Lp, SolverSelection::Auto),
            Ok(SolverChoice::QpIpm),
            "auto should route LP to the convex IPM (P=0)"
        );
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQp, SolverSelection::Auto),
            Ok(SolverChoice::QpIpm),
            "auto should route convex QP to the convex IPM"
        );
    }

    #[test]
    fn auto_routes_convex_qcqp_to_socp() {
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQcqp, SolverSelection::Auto),
            Ok(SolverChoice::SocpIpm),
            "auto should route convex QCQP to the conic IPM"
        );
    }

    #[test]
    fn auto_routes_nonconvex_to_nlp() {
        for class in [ProblemClass::NonconvexQp, ProblemClass::Nlp] {
            assert_eq!(
                resolve_solver(class, SolverSelection::Auto),
                Ok(SolverChoice::Nlp),
                "auto must resolve to Nlp for {:?}",
                class
            );
        }
    }

    #[test]
    fn forced_socp_accepts_convex_cone_family_only() {
        for class in [
            ProblemClass::Lp,
            ProblemClass::ConvexQp,
            ProblemClass::ConvexQcqp,
        ] {
            assert_eq!(
                resolve_solver(class, SolverSelection::Socp),
                Ok(SolverChoice::SocpIpm),
                "socp should accept {:?}",
                class
            );
        }
        assert!(resolve_solver(ProblemClass::NonconvexQp, SolverSelection::Socp).is_err());
        assert!(resolve_solver(ProblemClass::Nlp, SolverSelection::Socp).is_err());
    }

    #[test]
    fn forced_nlp_always_ok() {
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQp, SolverSelection::Nlp),
            Ok(SolverChoice::Nlp)
        );
    }

    #[test]
    fn forced_lp_on_nlp_errors() {
        let err = resolve_solver(ProblemClass::Nlp, SolverSelection::LpIpm).unwrap_err();
        assert!(err.contains("NLP"), "msg should name detected class: {err}");
        assert!(
            err.contains("lp-ipm"),
            "msg should name forced solver: {err}"
        );
    }

    #[test]
    fn forced_lp_on_lp_ok() {
        assert_eq!(
            resolve_solver(ProblemClass::Lp, SolverSelection::LpIpm),
            Ok(SolverChoice::LpIpm)
        );
    }

    #[test]
    fn forced_qp_accepts_lp_and_convex_qp_only() {
        assert_eq!(
            resolve_solver(ProblemClass::Lp, SolverSelection::QpIpm),
            Ok(SolverChoice::QpIpm)
        );
        assert_eq!(
            resolve_solver(ProblemClass::ConvexQp, SolverSelection::QpIpm),
            Ok(SolverChoice::QpIpm)
        );
        assert!(resolve_solver(ProblemClass::NonconvexQp, SolverSelection::QpIpm).is_err());
        assert!(resolve_solver(ProblemClass::Nlp, SolverSelection::QpIpm).is_err());
    }

    // --- Poly / quadratic analysis unit tests ---

    #[test]
    fn poly_of_quadratic_diagonal() {
        // (x0 - 1)^2  =>  x0^2 - 2 x0 + 1
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let h = analyze_quadratic(&e, 1).expect("degree-2 polynomial");
        // d²/dx0² (x0²) = 2
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
    }

    #[test]
    fn poly_rejects_transcendental() {
        // sin(x0) is not polynomial.
        let e = Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)));
        assert!(analyze_quadratic(&e, 1).is_none());
    }

    #[test]
    fn poly_rejects_cubic() {
        // x0^3
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        assert!(analyze_quadratic(&e, 1).is_none());
    }

    #[test]
    fn cross_term_hessian() {
        // x0 * x1  =>  H[0,1] = 1
        let e = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let h = analyze_quadratic(&e, 2).expect("degree-2");
        assert_eq!(h.get(&(0, 1)), Some(&1.0));
    }

    #[test]
    fn large_quadratic_sum_lowers_without_quadratic_blowup() {
        // Regression guard for the `solver_selection=auto` classifier hang
        // (mittelmann QCQP/bearing_400/qssp180 emitted zero iterations and
        // burned the full CPU budget). A quadratic expressed as a large
        // `Sum` of monomials must lower to a `Poly` in O(N log N): the old
        // `acc = acc.add(&to_poly(it)?)` ran the self-pruning `add` per
        // item, and `prune` rescans the whole accumulated map, so an
        // N-monomial sum was O(N²) and spun for >300 s before the solver
        // started (Ipopt solved the same problems in seconds). Build a
        // 5000-term sum of distinct squares and confirm the full diagonal
        // Hessian is recovered — this path completes effectively instantly
        // once the per-`add` prune is gone.
        const N: usize = 5000;
        let terms: Vec<Expr> = (0..N)
            .map(|i| Expr::Binary(BinOp::Mul, Box::new(Expr::Var(i)), Box::new(Expr::Var(i))))
            .collect();
        let e = Expr::Sum(terms);
        let h = analyze_quadratic(&e, N).expect("degree-2 sum of squares is a QP");
        assert_eq!(h.len(), N, "every xᵢ² contributes one diagonal entry");
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
        assert_eq!(h.get(&(N - 1, N - 1)), Some(&2.0));
    }

    // --- PSD test ---

    #[test]
    fn psd_accepts_convex_separable() {
        // diag(2, 4): both eigenvalues positive.
        let mut h = QuadHessian::new();
        h.insert((0, 0), 2.0);
        h.insert((1, 1), 4.0);
        assert!(hessian_is_psd(&h, 2));
    }

    #[test]
    fn psd_rejects_indefinite() {
        // [[0,1],[1,0]] has eigenvalues ±1.
        let mut h = QuadHessian::new();
        h.insert((0, 1), 1.0);
        assert!(!hessian_is_psd(&h, 2));
    }

    #[test]
    fn psd_accepts_psd_with_zero_eigenvalue() {
        // [[1,1],[1,1]] is PSD (eigenvalues 0 and 2).
        let mut h = QuadHessian::new();
        h.insert((0, 0), 1.0);
        h.insert((0, 1), 1.0);
        h.insert((1, 1), 1.0);
        assert!(hessian_is_psd(&h, 2));
    }

    // --- A1: ±PSD_TOL boundary of the convexity test (silent-misroute guard) ---

    /// The safety-critical case: a *real* negative direction — even a small
    /// one, well beyond `PSD_TOL` — must read non-PSD so an indefinite QP
    /// routes to NLP, never to the convex IPM (which would return a spurious
    /// "optimal" at a saddle/maximum).
    #[test]
    fn psd_rejects_small_but_real_negative_curvature() {
        // diag(2, −1e-3): min eigenvalue −1e-3 ≪ −PSD_TOL.
        let mut h = QuadHessian::new();
        h.insert((0, 0), 2.0);
        h.insert((1, 1), -1e-3);
        assert!(
            !hessian_is_psd(&h, 2),
            "a −1e-3 eigenvalue must read indefinite, not be rounded to PSD"
        );
    }

    /// Pin the threshold at exactly `±PSD_TOL` (1e-9). Within the band the
    /// test rounds a tiny negative eigenvalue to PSD **by design**: a
    /// genuinely semidefinite Hessian whose smallest eigenvalue computes as a
    /// tiny negative (Jacobi roundoff) must not be misread as nonconvex. The
    /// band is far below the error of solving a convex QP with that much
    /// curvature, so it is the sound tradeoff — see the A1 Finding in
    /// `dev-notes/pr70-hardening.md`. (1×1 Hessians are returned exactly, so
    /// this is deterministic.)
    #[test]
    fn psd_threshold_is_psd_tol() {
        let mut just_inside = QuadHessian::new();
        just_inside.insert((0, 0), -1e-10); // |λ| < PSD_TOL ⇒ treated as zero
        assert!(
            hessian_is_psd(&just_inside, 1),
            "−1e-10 is within tolerance and must round to PSD"
        );

        let mut just_outside = QuadHessian::new();
        just_outside.insert((0, 0), -1e-7); // |λ| > PSD_TOL ⇒ genuine negative
        assert!(
            !hessian_is_psd(&just_outside, 1),
            "−1e-7 is beyond tolerance and must read indefinite"
        );
    }

    // --- Sparse-factorization PSD certificate (CVXQP family) ---

    /// A large *diagonal* Hessian must take the O(nnz) sign fast path — no
    /// factorization at all — and read PSD. This is the large separable /
    /// least-squares QP shape (AUG2D, LISWET, …) that stays on the convex
    /// fast path.
    #[test]
    fn large_diagonal_hessian_is_cheap_and_psd() {
        let n = 50_000;
        let mut h = QuadHessian::new();
        for i in 0..n {
            h.insert((i, i), 2.0);
        }
        assert!(
            hessian_is_psd(&h, n),
            "diag(2,…,2) is PSD and must be settled by the O(nnz) sign path"
        );
    }

    /// A large *coupled* convex Hessian (off-diagonal terms over many
    /// variables) is the CVXQP-family shape that the old dense-Jacobi cap
    /// refused to certify (routing it to NLP). The sparse-factorization
    /// certificate now proves it PSD cheaply, so it reaches the convex
    /// solver. This is the regression fix.
    #[test]
    fn large_coupled_convex_hessian_is_certified_psd() {
        let k = 1_000;
        let mut h = QuadHessian::new();
        // Diagonally dominant tridiagonal: SPD. 2 on the diagonal, 0.1 on
        // the off-diagonal coupling chain ⇒ strictly diagonally dominant.
        for i in 0..k {
            h.insert((i, i), 2.0);
        }
        for i in 0..(k - 1) {
            h.insert((i, i + 1), 0.1);
        }
        assert!(
            hessian_is_psd(&h, k),
            "a diagonally-dominant coupled Hessian over {k} vars must be \
             certified PSD by the sparse factorization (CVXQP regression)"
        );
    }

    /// The sparse certificate must still *reject* a large coupled Hessian
    /// that is genuinely indefinite — size does not buy a free pass.
    #[test]
    fn large_coupled_indefinite_hessian_is_rejected() {
        let k = 1_000;
        let mut h = QuadHessian::new();
        for i in 0..k {
            h.insert((i, i), 2.0);
        }
        for i in 0..(k - 1) {
            h.insert((i, i + 1), 0.1);
        }
        // Flip one diagonal strongly negative ⇒ an indefinite direction.
        h.insert((0, 0), -5.0);
        assert!(
            !hessian_is_psd(&h, k),
            "a coupled Hessian with a strong negative-curvature direction \
             must be rejected regardless of size"
        );
    }

    /// A *small* coupled Hessian is certified by the same sparse path.
    #[test]
    fn small_coupled_hessian_is_certified_psd() {
        // [[2, 1], [1, 2]] — eigenvalues 1 and 3, PSD.
        let mut h = QuadHessian::new();
        h.insert((0, 0), 2.0);
        h.insert((0, 1), 1.0);
        h.insert((1, 1), 2.0);
        assert!(hessian_is_psd(&h, 2));
    }

    // --- End-to-end classify_problem on parsed .nl text ---

    /// Minimal `g`-format `.nl` text builder is overkill; instead use the
    /// reader's own fixtures via parse_nl_text on hand-written stubs.
    /// These cover the header LP fast-path and the AST walk.

    #[test]
    fn classify_pure_lp() {
        // minimize x0 + x1 s.t. x0 + x1 <= 1, no nonlinear parts.
        // Build an NlProblem directly for a hermetic test.
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0), (1, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![0.0, 0.0],
            x_u: vec![f64::INFINITY, f64::INFINITY],
            g_l: vec![f64::NEG_INFINITY],
            g_u: vec![1.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        assert_eq!(classify_problem(&prob), ProblemClass::Lp);
    }

    #[test]
    fn classify_convex_qp() {
        // minimize x0^2 + x1^2 s.t. linear; convex (H = diag(2,2)).
        let obj = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQp);
    }

    #[test]
    fn classify_nonconvex_qp() {
        // minimize x0 * x1 (indefinite Hessian) s.t. linear.
        let obj = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::NonconvexQp);
    }

    #[test]
    fn classify_nlp_from_transcendental_objective() {
        let obj = Expr::Unary(UnaryOp::Exp, Box::new(Expr::Var(0)));
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Regression: a `maximize` of a PSD-Hessian objective is a *concave*
    /// maximization ⇒ nonconvex minimization. The convexity test must run
    /// on the sense-adjusted Hessian, or this slips through to the convex
    /// IPM and returns a wrong (maximum/saddle) answer.
    #[test]
    fn classify_maximize_psd_objective_is_nonconvex() {
        // maximize x0^2 + x1^2 (H = diag(2,2), PSD) — concave max.
        let obj = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let mut prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        prob.minimize = false;
        assert_eq!(classify_problem(&prob), ProblemClass::NonconvexQp);
    }

    /// Mirror: `maximize` of a concave (NSD-Hessian) objective is a convex
    /// minimization once negated, so it is a legitimate `ConvexQp`.
    #[test]
    fn classify_maximize_concave_objective_is_convex() {
        // maximize −(x0^2 + x1^2) (H = diag(−2,−2)); negated ⇒ PSD.
        let neg_sq = |v: usize| {
            Expr::Unary(
                UnaryOp::Neg,
                Box::new(Expr::Binary(
                    BinOp::Pow,
                    Box::new(Expr::Var(v)),
                    Box::new(Expr::Const(2.0)),
                )),
            )
        };
        let obj = Expr::Binary(BinOp::Add, Box::new(neg_sq(0)), Box::new(neg_sq(1)));
        let mut prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        prob.minimize = false;
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQp);
    }

    #[test]
    fn classify_convex_qcqp() {
        // convex quadratic objective + a convex quadratic constraint.
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let prob = qp_stub(obj, vec![con]);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// Build a convex QCQP (linear objective + one convex quadratic
    /// constraint `x0² ≤ 1`) at an arbitrary declared `n`/`m`, padding the
    /// extra constraints with trivially-zero rows. Used to exercise the
    /// `SOCP_SIZE_BUDGET` routing cap without allocating `n×n` data.
    fn convex_qcqp_at_size(n: usize, m: usize) -> NlProblem {
        let mut con_nonlinear = vec![Expr::Const(0.0); m];
        con_nonlinear[0] = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let g_l = vec![f64::NEG_INFINITY; m];
        let mut g_u = vec![f64::INFINITY; m];
        g_u[0] = 1.0; // upper-only bound ⇒ convex feasible set
        NlProblem {
            n,
            m,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear,
            con_linear: vec![vec![]; m],
            x_l: vec![f64::NEG_INFINITY; n],
            x_u: vec![f64::INFINITY; n],
            g_l,
            g_u,
            x0: vec![0.0; n],
            lambda0: vec![0.0; m],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// A convex QCQP small enough to keep the conic path (n·m ≤ budget).
    #[test]
    fn small_convex_qcqp_routes_to_conic() {
        let prob = convex_qcqp_at_size(100, 100); // n·m = 1e4 ≪ budget
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// A convex QCQP whose `n·m` exceeds [`SOCP_SIZE_BUDGET`] falls back to
    /// NLP rather than the conic path — the mittelmann `nql180`/`qssp180`
    /// regression, where the O(n·m) SOCP reformulation burned the whole CPU
    /// budget before the solver started.
    #[test]
    fn oversized_convex_qcqp_falls_back_to_nlp() {
        // 10001 · 10001 ≈ 1.0002e8 > SOCP_SIZE_BUDGET (1e8).
        let prob = convex_qcqp_at_size(10_001, 10_001);
        assert!((prob.n as u64) * (prob.m as u64) > SOCP_SIZE_BUDGET);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Build a convex QCQP whose single quadratic constraint `(Σ xᵢ)² ≤ 1`
    /// couples all `k` variables (a dense rank-1 PSD Hessian over `k` vars),
    /// with `n = k`, `m = 1`. Exercises the per-row conic-reformulation guard
    /// independently of the `n·m` budget.
    fn coupled_convex_qcqp_with_k_vars(k: usize) -> NlProblem {
        // sum = x0 + x1 + … + x_{k-1}
        let mut sum = Expr::Var(0);
        for i in 1..k {
            sum = Expr::Binary(BinOp::Add, Box::new(sum), Box::new(Expr::Var(i)));
        }
        // constraint (Σ xᵢ)² ≤ 1 — convex feasible set, Hessian = 2·(all-ones),
        // PSD (rank 1) and fully coupled across all k variables.
        let con = Expr::Binary(BinOp::Pow, Box::new(sum), Box::new(Expr::Const(2.0)));
        NlProblem {
            n: k,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![con],
            con_linear: vec![vec![]],
            x_l: vec![f64::NEG_INFINITY; k],
            x_u: vec![f64::INFINITY; k],
            g_l: vec![f64::NEG_INFINITY],
            g_u: vec![1.0],
            x0: vec![0.0; k],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    /// A heavily-coupled *convex* QCQP constraint (here over 300 > 256 vars,
    /// `n·m = 300` well under [`SOCP_SIZE_BUDGET`]) must still fall back to NLP:
    /// the per-row SOC reformulation is `O(k³)` in the coupling width, which
    /// is the mittelmann `qcqp1000-*` hang (small `n·m`, ~1000-var coupled
    /// rows). The convexity certificate accepts it; the coupling guard routes
    /// it away from the conic path.
    #[test]
    fn heavily_coupled_convex_qcqp_falls_back_to_nlp() {
        let k = QCQP_SOCP_COUPLED_VARS + 44; // 300
        let prob = coupled_convex_qcqp_with_k_vars(k);
        assert!((prob.n as u64) * (prob.m as u64) <= SOCP_SIZE_BUDGET);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// The companion to the guard: a convex QCQP whose constraint couples few
    /// enough variables keeps the conic path. Same `(Σ xᵢ)² ≤ 1` shape over
    /// `k ≤ QCQP_SOCP_COUPLED_VARS` vars ⇒ `ConvexQcqp`.
    #[test]
    fn lightly_coupled_convex_qcqp_keeps_conic() {
        let k = QCQP_SOCP_COUPLED_VARS - 6; // 250 ≤ 256
        let prob = coupled_convex_qcqp_with_k_vars(k);
        assert_eq!(classify_problem(&prob), ProblemClass::ConvexQcqp);
    }

    /// Classification mirror of the boundary guard: a QP whose only
    /// curvature is a genuine (beyond-tolerance) negative direction is
    /// `NonconvexQp`, so `auto` routes it to NLP rather than the convex IPM.
    /// `minimize −x0²` is concave for a minimizer ⇒ indefinite.
    #[test]
    fn classify_concave_minimize_is_nonconvex() {
        let obj = Expr::Unary(
            UnaryOp::Neg,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::NonconvexQp);
    }

    /// Conservative QCQP guard: a convex quadratic objective with an
    /// *indefinite* quadratic constraint must fall back to NLP — never be
    /// called `ConvexQcqp` and handed to the conic path, which would treat a
    /// nonconvex feasible region as convex.
    #[test]
    fn classify_qcqp_with_indefinite_constraint_falls_back_to_nlp() {
        // obj x0² (convex); constraint x0·x1 (indefinite Hessian).
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let prob = qp_stub(obj, vec![con]);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Sense guard: a PSD-Hessian quadratic constraint is convex only as an
    /// **upper** bound. With a finite *lower* bound (`g(x) ≥ g_l`) the
    /// feasible set is the nonconvex super-level set, so it must fall back to
    /// NLP — never be routed to the conic solver as if convex.
    #[test]
    fn classify_psd_quadratic_with_lower_bound_is_nonconvex() {
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let mut prob = qp_stub(obj, vec![con]);
        // g(x) ≥ 1  (finite lower, infinite upper) — convex function, but the
        // ≥ side is a nonconvex region.
        prob.g_l = vec![1.0];
        prob.g_u = vec![f64::INFINITY];
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Sense guard: a quadratic *equality* (`g(x) = c`) is nonconvex even
    /// with a PSD Hessian, so it must fall back to NLP, not ConvexQcqp.
    #[test]
    fn classify_quadratic_equality_is_nonconvex() {
        let obj = Expr::Const(0.0);
        let con = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let mut prob = qp_stub(obj, vec![con]);
        prob.g_l = vec![1.0];
        prob.g_u = vec![1.0]; // x0² = 1 — nonconvex.
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// A nonlinear objective expression whose quadratic part algebraically
    /// cancels has an empty Hessian ⇒ classify as `Lp`, not a spurious QP
    /// (which would otherwise route a linear problem to the QP IPM).
    #[test]
    fn classify_cancelling_quadratic_objective_is_lp() {
        // x0² − x0²  ≡ 0: the degree-2 terms cancel in the polynomial walk.
        let sq = || {
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(2.0)),
            )
        };
        let obj = Expr::Binary(BinOp::Sub, Box::new(sq()), Box::new(sq()));
        let prob = qp_stub(obj, vec![Expr::Const(0.0)]);
        assert_eq!(classify_problem(&prob), ProblemClass::Lp);
    }

    #[test]
    fn classify_nlp_from_transcendental_constraint() {
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(2.0)),
        );
        let con = Expr::Unary(UnaryOp::Log, Box::new(Expr::Var(1)));
        let prob = qp_stub(obj, vec![con]);
        assert_eq!(classify_problem(&prob), ProblemClass::Nlp);
    }

    /// Build a 2-var, 1-con problem stub with the given nonlinear
    /// objective and per-constraint nonlinear parts. Linear parts and
    /// bounds are filled with benign defaults.
    fn qp_stub(obj_nonlinear: Expr, con_nonlinear: Vec<Expr>) -> NlProblem {
        let m = con_nonlinear.len();
        NlProblem {
            n: 2,
            m,
            num_obj: 1,
            minimize: true,
            obj_nonlinear,
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear,
            con_linear: vec![vec![]; m],
            x_l: vec![f64::NEG_INFINITY; 2],
            x_u: vec![f64::INFINITY; 2],
            g_l: vec![f64::NEG_INFINITY; m],
            g_u: vec![0.0; m],
            x0: vec![0.0; 2],
            lambda0: vec![0.0; m],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    // Keep parse_nl_text reachable for a future header-fast-path test
    // against a committed .nl fixture.
    #[allow(dead_code)]
    fn _parse(txt: &str) -> NlProblem {
        parse_nl_text(txt).expect("valid .nl")
    }
}
