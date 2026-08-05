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
//! # What it refuses
//!
//! Everything the walker refuses (transcendentals, external calls, `x^y`,
//! division by a non-constant, degree past the cap), plus — for the SOS slice
//! specifically — a `maximize` objective and any problem carrying constraints
//! or finite variable bounds. The last restriction is not a limitation of the
//! mathematics but of what v1 *proves*: `global_lower_bound_of_sos` gives
//! `γ ≤ p(x)` for every `x ∈ ℝⁿ`. On a constrained problem that statement is
//! still true, but it is a bound on the wrong problem — the constrained minimum
//! can be strictly larger — so emitting it would answer a question nobody asked.
//! The Putinar form that handles constraints needs several SOS blocks and is
//! deferred with its consumer.

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
    if prob.m != 0 {
        return Err(format!(
            "problem has {} constraint(s); the SOS slice (v1) certifies a global \
             bound over all of ℝⁿ, which would not be a bound on the constrained \
             minimum",
            prob.m
        ));
    }
    for i in 0..n {
        if lower_bound_present(prob.x_l[i]) || upper_bound_present(prob.x_u[i]) {
            return Err(format!(
                "variable {i} has a finite bound; the SOS slice (v1) certifies a \
                 global bound over all of ℝⁿ (see the constraint refusal above)"
            ));
        }
    }

    let poly = to_poly_bounded(&prob.obj_nonlinear, MAX_POLY_DEGREE).ok_or(
        "objective is not a polynomial (or exceeds the degree cap); SOS \
         certification needs a polynomial objective",
    )?;

    // Multiset-of-indices → exponent vector, merging like monomials.
    let mut terms: std::collections::BTreeMap<Vec<usize>, f64> = std::collections::BTreeMap::new();
    for (monomial, coeff) in &poly.terms {
        let mut exps = vec![0usize; n];
        for &i in monomial {
            *exps
                .get_mut(i)
                .ok_or_else(|| format!("objective references variable {i} but n = {n}"))? += 1;
        }
        *terms.entry(exps).or_insert(0.0) += coeff;
    }
    // The `.nl` linear part and constant live outside the tree; fold them in.
    for &(i, v) in &prob.obj_linear {
        let mut exps = vec![0usize; n];
        *exps
            .get_mut(i)
            .ok_or_else(|| format!("objective references variable {i} but n = {n}"))? = 1;
        *terms.entry(exps).or_insert(0.0) += v;
    }
    if prob.obj_constant != 0.0 {
        *terms.entry(vec![0usize; n]).or_insert(0.0) += prob.obj_constant;
    }

    // Drop terms that cancelled to exactly zero. This is not cosmetic: a zero
    // coefficient reaches the coefficient-matching system as a real equation,
    // and an all-zero monomial the basis cannot reach makes `round_gram` report
    // `Inconsistent` for a polynomial that is in fact certifiable.
    terms.retain(|_, c| *c != 0.0);

    Ok(PolyObjective {
        n,
        terms: terms.into_iter().collect(),
    })
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

    /// The refusal that matters most: on a constrained problem the emitted
    /// bound would be true but about the wrong problem.
    #[test]
    fn a_bounded_variable_is_refused() {
        let mut p = prob(1, pow(Expr::Var(0), 4.0));
        p.x_l = vec![0.0];
        let err = extract_poly_objective(&p).unwrap_err();
        assert!(err.contains("finite bound"), "got {err}");
    }
}
