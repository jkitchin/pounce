//! Extract a polynomial objective from a parsed `.nl`, for the SOS
//! certification path (`verdict = global-lower-bound`).
//!
//! This is the degree-general sibling of [`crate::qp_extract`]. Both read the
//! same objective through the same walker ([`crate::dispatch::to_poly_bounded`]);
//! they differ only in what they do with the result — `qp_extract` demands
//! degree ≤ 2 and builds a Hessian, this one keeps every monomial.
//!
//! # Representation
//!
//! The walker returns each monomial as a *sorted multiset of variable indices*
//! (`[0,0,0,0]` is `x₀⁴`); the certificate wants an *exponent vector* of length
//! `n` (`[4]`). Converting between them is this module's job, and it is the only
//! place the two spellings meet.
//!
//! # Constraints
//!
//! A constrained problem extracts as a *feasible set*: a list of polynomial
//! `gₖ(x) ≥ 0`, which the Putinar path certifies a bound over. Three sources
//! feed it, and all three are ordinary inequalities once normalized —
//! `lower ≤ body`, `body ≤ upper`, and each finite variable bound. A range
//! constraint contributes both halves.
//!
//! # What it refuses
//!
//! Everything the walker refuses (transcendentals, external calls, `x^y`,
//! division by a non-constant, degree past the cap), plus — for the SOS slice
//! specifically — a `maximize` objective and any *equality* constraint.
//!
//! The equality refusal is about what is provable, not what is expressible. A
//! Positivstellensatz certificate handles `h(x) = 0` with a **free** multiplier
//! `λ(x)` of either sign, not an SOS one, and the consumer's
//! `constrained_lower_bound_of_sos` takes only the SOS form. Emitting an
//! equality as two inequalities would be sound but is a trap in practice: the
//! resulting feasible set has empty interior, which is exactly where Putinar's
//! theorem stops guaranteeing that a certificate exists at any degree.

use pounce_common::types::{lower_bound_present, upper_bound_present};
use pounce_nl::nl_reader::NlProblem;

use crate::dispatch::{MAX_POLY_DEGREE, to_poly_bounded};

/// A polynomial objective as `(exponent vector, coefficient)` pairs, with the
/// exponent vectors of length `n_vars`. Sorted by exponent vector, so the same
/// objective always extracts to the same list.
#[derive(Clone, Debug, PartialEq)]
pub struct PolyObjective {
    pub n: usize,
    pub terms: Vec<(Vec<usize>, f64)>,
}

impl PolyObjective {
    /// Total degree of the polynomial (0 for the empty/zero polynomial).
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(e, _)| e.iter().sum::<usize>())
            .max()
            .unwrap_or(0)
    }
}

/// Extract the objective of `prob` as a polynomial term list, or explain why it
/// is off the SOS slice.
///
/// The `.nl` splits an objective across three places — a linear part, a
/// constant, and the nonlinear expression tree — and *all three* contribute.
/// Reading only the tree silently drops the linear terms of, say,
/// `x⁴ − 2x² + 3x`, producing a certificate for a different polynomial than the
/// file describes. (`qp_extract` folds the same three sources for the same
/// reason.)
pub fn extract_poly_objective(prob: &NlProblem) -> Result<PolyObjective, String> {
    let n = prob.n;
    if !prob.minimize {
        return Err("SOS certification supports minimize objectives only (v1)".to_string());
    }
    let terms = poly_terms(
        n,
        &prob.obj_nonlinear,
        &prob.obj_linear,
        prob.obj_constant,
        "objective",
    )?;
    Ok(PolyObjective { n, terms })
}

/// A polynomial in the `.nl`'s three-part form — nonlinear tree, linear part,
/// constant — as a single term list over exponent vectors.
///
/// *All three* contribute. Reading only the tree silently drops the linear terms
/// of, say, `x⁴ − 2x² + 3x`, producing a certificate for a different polynomial
/// than the file describes. (`qp_extract` folds the same three sources for the
/// same reason.) `what` names the source in error messages.
fn poly_terms(
    n: usize,
    tree: &pounce_nl::nl_reader::Expr,
    linear: &[(usize, f64)],
    constant: f64,
    what: &str,
) -> Result<Vec<(Vec<usize>, f64)>, String> {
    let poly = to_poly_bounded(tree, MAX_POLY_DEGREE).ok_or_else(|| {
        format!(
            "{what} is not a polynomial (or exceeds the degree cap); SOS \
             certification needs polynomial data"
        )
    })?;

    // Multiset-of-indices → exponent vector, merging like monomials.
    let mut terms: std::collections::BTreeMap<Vec<usize>, f64> = std::collections::BTreeMap::new();
    for (monomial, coeff) in &poly.terms {
        let mut exps = vec![0usize; n];
        for &i in monomial {
            *exps
                .get_mut(i)
                .ok_or_else(|| format!("{what} references variable {i} but n = {n}"))? += 1;
        }
        *terms.entry(exps).or_insert(0.0) += coeff;
    }
    for &(i, v) in linear {
        let mut exps = vec![0usize; n];
        *exps
            .get_mut(i)
            .ok_or_else(|| format!("{what} references variable {i} but n = {n}"))? = 1;
        *terms.entry(exps).or_insert(0.0) += v;
    }
    if constant != 0.0 {
        *terms.entry(vec![0usize; n]).or_insert(0.0) += constant;
    }

    // Drop terms that cancelled to exactly zero. This is not cosmetic: a zero
    // coefficient reaches the coefficient-matching system as a real equation,
    // and an all-zero monomial the basis cannot reach makes `round_gram` report
    // `Inconsistent` for a polynomial that is in fact certifiable.
    terms.retain(|_, c| *c != 0.0);

    Ok(terms.into_iter().collect())
}

/// Extract the feasible set of `prob` as polynomial `gₖ(x) ≥ 0` term lists, or
/// explain why it is off the SOS slice. Empty means unconstrained — a bound over
/// all of `ℝⁿ`.
///
/// Every source of an inequality is normalized to the same `g ≥ 0` form: a row
/// `lower ≤ body` becomes `body − lower`, a row `body ≤ upper` becomes
/// `upper − body`, and a range row contributes both. Finite variable bounds are
/// rows like any other here (`xⱼ − l`, `u − xⱼ`) — the Putinar certificate has
/// no notion of a "bound", and folding them in is what lets it bound an
/// objective that is unbounded on `ℝⁿ`.
///
/// The order is fixed: constraint rows in file order (lower half before upper
/// half of a range), then variable bounds by index. That order is the
/// certificate's `multiplier` indices, so it must reproduce exactly on the
/// consumer's re-derivation.
pub fn extract_poly_constraints(prob: &NlProblem) -> Result<Vec<Vec<(Vec<usize>, f64)>>, String> {
    let n = prob.n;
    let mut out: Vec<Vec<(Vec<usize>, f64)>> = Vec::new();

    for i in 0..prob.m {
        let (lo, hi) = (prob.g_l[i], prob.g_u[i]);
        let (lo_present, hi_present) = (lower_bound_present(lo), upper_bound_present(hi));
        if lo_present && hi_present && lo == hi {
            return Err(format!(
                "constraint {i} is an equality; the SOS slice certifies a bound over \
                 a feasible set cut out by inequalities. An equality needs a free \
                 (sign-unrestricted) multiplier, which the consumer's \
                 Positivstellensatz theorem does not take"
            ));
        }
        if !lo_present && !hi_present {
            continue; // a free row constrains nothing
        }
        let body = poly_terms(
            n,
            &prob.con_nonlinear[i],
            &prob.con_linear[i],
            0.0,
            &format!("constraint {i}"),
        )?;
        if lo_present {
            out.push(shifted(&body, -lo, 1.0, n));
        }
        if hi_present {
            out.push(shifted(&body, hi, -1.0, n));
        }
    }

    for j in 0..n {
        if lower_bound_present(prob.x_l[j]) {
            out.push(shifted(&unit(j, n), -prob.x_l[j], 1.0, n));
        }
        if upper_bound_present(prob.x_u[j]) {
            out.push(shifted(&unit(j, n), prob.x_u[j], -1.0, n));
        }
    }

    if out.iter().any(|g| g.is_empty()) {
        return Err(
            "a constraint reduced to the zero polynomial, which is vacuous \
                    (0 ≥ 0) and leaves the relaxation with a block that proves nothing"
                .to_string(),
        );
    }
    Ok(out)
}

/// The term list of the bare variable `xⱼ`.
fn unit(j: usize, n: usize) -> Vec<(Vec<usize>, f64)> {
    let mut e = vec![0usize; n];
    e[j] = 1;
    vec![(e, 1.0)]
}

/// `constant + sign·body`, as a term list with exact-zero terms dropped.
fn shifted(
    body: &[(Vec<usize>, f64)],
    constant: f64,
    sign: f64,
    n: usize,
) -> Vec<(Vec<usize>, f64)> {
    let mut terms: std::collections::BTreeMap<Vec<usize>, f64> = std::collections::BTreeMap::new();
    for (e, c) in body {
        *terms.entry(e.clone()).or_insert(0.0) += sign * c;
    }
    if constant != 0.0 {
        *terms.entry(vec![0usize; n]).or_insert(0.0) += constant;
    }
    terms.retain(|_, c| *c != 0.0);
    terms.into_iter().collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use pounce_nl::nl_reader::{BinOp, Expr, UnaryOp};

    /// A bare `NlProblem`: `n` free (unbounded) variables, no constraints, and
    /// the given objective tree. The unconstrained shape the SOS slice accepts.
    fn prob(n: usize, obj: Expr) -> NlProblem {
        NlProblem {
            n,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: obj,
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![f64::NEG_INFINITY; n],
            x_u: vec![f64::INFINITY; n],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0; n],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        }
    }

    fn pow(base: Expr, k: f64) -> Expr {
        Expr::Binary(BinOp::Pow, Box::new(base), Box::new(Expr::Const(k)))
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn sub(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b))
    }

    /// `x⁴ − 2x² + 2` — the reference quartic, degree 4, one variable.
    #[test]
    fn quartic_extracts_to_its_term_list() {
        let obj = Expr::Binary(
            BinOp::Add,
            Box::new(sub(
                pow(Expr::Var(0), 4.0),
                mul(Expr::Const(2.0), pow(Expr::Var(0), 2.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let p = extract_poly_objective(&prob(1, obj)).unwrap();
        assert_eq!(p.degree(), 4);
        assert_eq!(
            p.terms,
            vec![(vec![0], 2.0), (vec![2], -2.0), (vec![4], 1.0)]
        );
    }

    /// Degree 4 is past `to_poly`'s degree-2 cap, so this is the case that
    /// would have been refused before the walker took a bound.
    #[test]
    fn multivariate_cross_terms_get_per_variable_exponents() {
        // (x₀·x₁)² = x₀²x₁²
        let obj = pow(mul(Expr::Var(0), Expr::Var(1)), 2.0);
        let p = extract_poly_objective(&prob(2, obj)).unwrap();
        assert_eq!(p.terms, vec![(vec![2, 2], 1.0)]);
    }

    /// The `.nl` keeps linear terms out of the tree. Dropping them would
    /// certify a bound for a *different* polynomial — silently, since the
    /// result still looks like a valid certificate.
    #[test]
    fn linear_part_and_constant_outside_the_tree_are_folded_in() {
        let mut p = prob(1, pow(Expr::Var(0), 4.0));
        p.obj_linear = vec![(0, 3.0)];
        p.obj_constant = 5.0;
        let out = extract_poly_objective(&p).unwrap();
        assert_eq!(
            out.terms,
            vec![(vec![0], 5.0), (vec![1], 3.0), (vec![4], 1.0)]
        );
    }

    /// A term that cancels to exactly zero must not survive as `0·x²`.
    #[test]
    fn cancelled_terms_are_dropped() {
        // x² − x²
        let obj = sub(pow(Expr::Var(0), 2.0), pow(Expr::Var(0), 2.0));
        let out = extract_poly_objective(&prob(1, obj)).unwrap();
        assert!(out.terms.is_empty(), "got {:?}", out.terms);
    }

    #[test]
    fn transcendental_objective_is_refused() {
        let obj = Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)));
        assert!(extract_poly_objective(&prob(1, obj)).is_err());
    }

    #[test]
    fn fractional_exponent_is_refused() {
        let obj = pow(Expr::Var(0), 2.5);
        assert!(extract_poly_objective(&prob(1, obj)).is_err());
    }

    #[test]
    fn maximize_is_refused() {
        let mut p = prob(1, pow(Expr::Var(0), 4.0));
        p.minimize = false;
        assert!(extract_poly_objective(&p).is_err());
    }

    #[test]
    fn a_free_problem_has_an_empty_feasible_set_description() {
        let p = prob(1, pow(Expr::Var(0), 4.0));
        assert!(extract_poly_constraints(&p).unwrap().is_empty());
    }

    /// A box on `x` is two Putinar constraints, `x − l ≥ 0` and `u − x ≥ 0`.
    /// This is the case that makes the constrained path worth having: it lets
    /// a bound exist for an objective that is unbounded below off the box.
    #[test]
    fn variable_bounds_become_constraints() {
        let mut p = prob(1, pow(Expr::Var(0), 3.0));
        p.x_l = vec![-1.0];
        p.x_u = vec![2.0];
        assert_eq!(
            extract_poly_constraints(&p).unwrap(),
            vec![
                vec![(vec![0], 1.0), (vec![1], 1.0)],  // x + 1
                vec![(vec![0], 2.0), (vec![1], -1.0)], // 2 − x
            ]
        );
    }

    /// A one-sided row gives one constraint; a range row gives two, lower half
    /// first. That order is the certificate's `multiplier` indices.
    #[test]
    fn constraint_rows_come_out_in_multiplier_order() {
        let mut p = prob(2, pow(Expr::Var(0), 4.0));
        p.m = 2;
        // row 0: x₀x₁ ≥ 1   row 1: 3 ≤ x₀² ≤ 5
        p.con_nonlinear = vec![mul(Expr::Var(0), Expr::Var(1)), pow(Expr::Var(0), 2.0)];
        p.con_linear = vec![vec![], vec![]];
        p.g_l = vec![1.0, 3.0];
        p.g_u = vec![f64::INFINITY, 5.0];
        assert_eq!(
            extract_poly_constraints(&p).unwrap(),
            vec![
                vec![(vec![0, 0], -1.0), (vec![1, 1], 1.0)], // x₀x₁ − 1
                vec![(vec![0, 0], -3.0), (vec![2, 0], 1.0)], // x₀² − 3
                vec![(vec![0, 0], 5.0), (vec![2, 0], -1.0)], // 5 − x₀²
            ]
        );
    }

    /// An equality would need a sign-unrestricted multiplier, not an SOS one.
    #[test]
    fn an_equality_constraint_is_refused() {
        let mut p = prob(1, pow(Expr::Var(0), 4.0));
        p.m = 1;
        p.con_nonlinear = vec![pow(Expr::Var(0), 2.0)];
        p.con_linear = vec![vec![]];
        p.g_l = vec![1.0];
        p.g_u = vec![1.0];
        let err = extract_poly_constraints(&p).unwrap_err();
        assert!(err.contains("equality"), "got {err}");
    }

    /// A nonpolynomial *constraint* is refused just as a nonpolynomial
    /// objective is — and the message must name the row, not the objective.
    #[test]
    fn a_transcendental_constraint_is_refused() {
        let mut p = prob(1, pow(Expr::Var(0), 4.0));
        p.m = 1;
        p.con_nonlinear = vec![Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)))];
        p.con_linear = vec![vec![]];
        p.g_l = vec![0.0];
        p.g_u = vec![f64::INFINITY];
        let err = extract_poly_constraints(&p).unwrap_err();
        assert!(err.contains("constraint 0"), "got {err}");
    }
}
