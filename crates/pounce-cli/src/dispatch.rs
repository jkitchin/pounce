//! Solver routing (Phase 1 of the LP/QP dispatch plan).
//!
//! See `dev-notes/lp-qp-routing.md`. This module sits between problem
//! loading and the call to `optimize_tnlp`. It does three things:
//!
//! 1. **Classify** the parsed problem into a [`ProblemClass`] by walking
//!    the nonlinear expression trees the `.nl` reader already produced.
//! 2. **Resolve** that class against the user's `solver_selection`
//!    option into a [`SolverChoice`].
//! 3. (Phase 2+) **Dispatch** to the chosen solver.
//!
//! Phase 1 ships with *no behavior change*: the only solvers wired are
//! `Nlp` (the existing filter-IPM) and `auto`, which resolves to `Nlp`
//! for every class until `pounce-convex` lands. The classifier and the
//! option plumbing are fully present and tested so Phase 2 can drop in
//! the specialized solvers behind the seam.
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
/// in this path; for Phase 1 a fixed absolute tolerance is adequate and
/// errs toward the safe (more general) class.
const PSD_TOL: f64 = 1e-9;

/// The mathematical class of a loaded problem, from most to least
/// specialized. See the module docs and `dev-notes/lp-qp-routing.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemClass {
    /// Linear objective, linear constraints.
    Lp,
    /// Convex quadratic objective, linear constraints (Hessian PSD).
    ConvexQp,
    /// Convex quadratic objective and/or convex quadratic constraints.
    /// SOCP-representable; routes to the conic solver from Phase 4.
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
/// Phase 1 only ever resolves to [`SolverChoice::Nlp`]; the other
/// variants exist so the option parser and the forced-selection
/// validation are complete, and so Phase 2 can wire them without
/// touching this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverChoice {
    /// The existing Wächter-Biegler filter-IPM. The only solver wired in
    /// Phase 1.
    Nlp,
    /// IPM-LP in `pounce-convex` (Phase 2).
    LpIpm,
    /// IPM-QP in `pounce-convex` (Phase 2).
    QpIpm,
    /// Active-set QP in `pounce-qp` (parallel track).
    QpActiveSet,
    /// Spatial branch-and-bound global optimizer (`pounce-global`).
    Global,
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
    /// Force active-set QP; error if the problem is not LP/convex-QP.
    QpActiveSet,
    /// Force the spatial branch-and-bound global solver (any class).
    Global,
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
            "qp-active-set" => Some(SolverSelection::QpActiveSet),
            "global" => Some(SolverSelection::Global),
            _ => None,
        }
    }

    /// The accepted values, for error messages and option registration.
    pub const VALUES: &'static [&'static str] =
        &["auto", "nlp", "lp-ipm", "qp-ipm", "qp-active-set", "global"];
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
        // Convex QCQP requires every ≤-inequality's constraint Hessian
        // to be PSD. Phase 1 does not yet distinguish constraint sense /
        // curvature sign per row with full rigor, so be conservative:
        // only call it ConvexQcqp when every quadratic constraint's
        // Hessian is PSD; otherwise fall back to NLP (sound: NLP-IPM
        // finds a local min either way).
        for c in &prob.con_nonlinear {
            if is_trivially_zero(c) {
                continue;
            }
            match analyze_quadratic(c, prob.n) {
                Some(q) if q.is_empty() => {}
                Some(q) => {
                    if !hessian_is_psd(&q, prob.n) {
                        return ProblemClass::Nlp;
                    }
                }
                None => return ProblemClass::Nlp,
            }
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
/// In Phase 1 the resolved choice is informational for everything except
/// `Nlp`: the dispatcher (Phase 2) is what acts on `LpIpm` / `QpIpm` /
/// `QpActiveSet`. `auto` resolves to `Nlp` for every class until
/// `pounce-convex` lands (documented no-op so there is no regression).
pub fn resolve_solver(
    class: ProblemClass,
    selection: SolverSelection,
) -> Result<SolverChoice, String> {
    use ProblemClass as P;
    use SolverSelection as S;

    // Is this class within the convex-QP family (LP or convex QP)?
    let is_lp = class == P::Lp;
    let is_convex_qp = matches!(class, P::Lp | P::ConvexQp);

    match selection {
        // `auto`: route LP and convex QP to the specialized convex IPM
        // (`pounce-convex`); everything else (QCQP until the conic
        // solver lands, nonconvex QP, general NLP) falls through to the
        // NLP filter-IPM. LP is solved by the same QP IPM (P = 0), so it
        // resolves to `QpIpm` rather than a distinct LP entry point.
        S::Auto => match class {
            P::Lp | P::ConvexQp => Ok(SolverChoice::QpIpm),
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
        S::QpActiveSet => {
            if is_convex_qp {
                Ok(SolverChoice::QpActiveSet)
            } else {
                Err(mismatch_msg(class, "qp-active-set", "an LP or convex QP"))
            }
        }
        // The global solver handles any factorable class (it just needs finite
        // variable bounds); `auto` never selects it, so this is opt-in only.
        S::Global => Ok(SolverChoice::Global),
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

/// Attempt to read an expression as a polynomial of total degree ≤ 2 and
/// return its Hessian (constant, since the form is quadratic). Returns
/// `None` if the expression contains any term the classifier cannot
/// prove is degree-≤2 polynomial (transcendental ops, division by a
/// non-constant, `Pow` with exponent ∉ {0,1,2}, products of degree > 2,
/// external calls, …). `None` ⇒ treat as general nonlinear.
pub(crate) fn analyze_quadratic(e: &Expr, n: usize) -> Option<QuadHessian> {
    analyze_quadratic_full(e, n).map(|(h, _)| h)
}

/// Like [`analyze_quadratic`] but also returns the degree-1 (linear)
/// coefficients of the form: `(Hessian, [(var, coef), …])`.
///
/// AMPL folds the linear part of a nonlinear term into the objective's
/// nonlinear expression tree (the `−6·x₀` of `(x₀−3)²`, say) rather than
/// the linear section. Callers building the QP objective vector `c` must
/// add these in, exactly as the NLP path's `eval_f` sums the linear
/// section *and* the nonlinear tree — otherwise the linear shift is
/// silently dropped and the convex solve minimizes the wrong objective.
pub(crate) fn analyze_quadratic_full(
    e: &Expr,
    _n: usize,
) -> Option<(QuadHessian, Vec<(usize, f64)>)> {
    let poly = to_poly(e)?;
    if poly.max_degree() > 2 {
        return None;
    }
    let mut h: QuadHessian = BTreeMap::new();
    let mut lin: Vec<(usize, f64)> = Vec::new();
    for (vars, coef) in &poly.terms {
        match vars.as_slice() {
            // Constant term contributes nothing to gradient or Hessian.
            [] => {}
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
    Some((h, lin))
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
            let mut acc = Poly::default();
            for it in items {
                acc = acc.add(&to_poly(it)?);
            }
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

/// Is the (symmetric, sparse) Hessian positive semidefinite?
///
/// Builds the dense symmetric matrix over the variables that actually
/// appear in the quadratic form and runs a symmetric eigenvalue check
/// via Jacobi rotations — adequate for the small-to-moderate dense
/// blocks a classifier sees, and dependency-free. Returns `true` only
/// when the smallest eigenvalue is `≥ -PSD_TOL`; an inconclusive or
/// clearly-negative result returns `false`, routing to the safe
/// (more general) class.
fn hessian_is_psd(h: &QuadHessian, _n: usize) -> bool {
    if h.is_empty() {
        return true; // zero matrix is PSD (the linear case)
    }
    // Compress to the active variable set so the dense matrix is small.
    let mut active: Vec<usize> = Vec::new();
    for (i, j) in h.keys() {
        active.push(*i);
        active.push(*j);
    }
    active.sort_unstable();
    active.dedup();
    let k = active.len();
    let idx = |v: usize| active.binary_search(&v).unwrap();

    let mut a = vec![0.0f64; k * k];
    for ((i, j), v) in h {
        let (ri, rj) = (idx(*i), idx(*j));
        a[ri * k + rj] = *v;
        a[rj * k + ri] = *v;
    }

    match smallest_eigenvalue_symmetric(&mut a, k) {
        Some(min_eig) => min_eig >= -PSD_TOL,
        None => false, // did not converge ⇒ be conservative
    }
}

/// Smallest eigenvalue of a dense `k×k` symmetric matrix (row-major) via
/// the classical cyclic Jacobi eigenvalue algorithm. Destroys `a`.
/// Returns `None` if it fails to converge within the sweep budget.
fn smallest_eigenvalue_symmetric(a: &mut [f64], k: usize) -> Option<f64> {
    if k == 0 {
        return Some(0.0);
    }
    if k == 1 {
        return Some(a[0]);
    }
    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off += a[p * k + q] * a[p * k + q];
            }
        }
        if off <= 1e-30 {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[p * k + p];
                let aqq = a[q * k + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Apply the rotation J^T A J.
                for i in 0..k {
                    let aip = a[i * k + p];
                    let aiq = a[i * k + q];
                    a[i * k + p] = c * aip - s * aiq;
                    a[i * k + q] = s * aip + c * aiq;
                }
                for i in 0..k {
                    let api = a[p * k + i];
                    let aqi = a[q * k + i];
                    a[p * k + i] = c * api - s * aqi;
                    a[q * k + i] = s * api + c * aqi;
                }
            }
        }
    }
    let mut min_eig = f64::INFINITY;
    for i in 0..k {
        min_eig = min_eig.min(a[i * k + i]);
    }
    if min_eig.is_finite() {
        Some(min_eig)
    } else {
        None
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
    fn auto_routes_everything_else_to_nlp() {
        for class in [
            ProblemClass::ConvexQcqp, // until the conic solver lands
            ProblemClass::NonconvexQp,
            ProblemClass::Nlp,
        ] {
            assert_eq!(
                resolve_solver(class, SolverSelection::Auto),
                Ok(SolverChoice::Nlp),
                "auto must resolve to Nlp for {:?}",
                class
            );
        }
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
