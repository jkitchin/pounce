//! Coupling classification for candidate auxiliary blocks.
//!
//! PR 5 of the auxiliary-presolve port (issue #53). After PRs 2–4
//! find independent square blocks, this module decides whether each
//! one is **safe** to eliminate.
//!
//! A block is unsafe if it interacts with parts of the problem the
//! IPM still owns:
//!
//! - **Inequality coupling** — a block variable appears in any
//!   inequality row. Eliminating the block would fix that variable's
//!   value, possibly violating the inequality.
//! - **Objective coupling** — a block variable contributes to the
//!   objective gradient. Eliminating fixes the value and removes the
//!   IPM's freedom to move that variable to improve the objective.
//!
//! ripopt anchor: `src/auxiliary_preprocessing.rs:39-59, 1642-1687`.

use std::collections::HashSet;

use pounce_common::types::Number;

use crate::btf::BlockTriangularBlock;
use crate::incidence::InequalityIncidence;

/// How a candidate block is coupled to the rest of the problem.
///
/// Drives the elimination policy: `PureEquality` is always eligible
/// under `AuxiliaryCouplingPolicy::Safe`, `ObjectiveCoupled` adds in
/// under `Aggressive`, and the two inequality-coupled variants are
/// never eliminated in v1 (matches ripopt's conservative default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxiliaryCouplingClass {
    /// Block touches only equality rows and not the objective.
    PureEquality,
    /// Block variables appear in the objective gradient.
    ObjectiveCoupled,
    /// Block variables appear in at least one inequality row.
    InequalityCoupled,
    /// Both of the above.
    ObjectiveAndInequalityCoupled,
}

/// Return the set of variable indices where the objective gradient
/// is non-negligible (`|grad_f[i]| > zero_tol`).
pub fn objective_gradient_support(grad_f: &[Number], zero_tol: Number) -> HashSet<usize> {
    grad_f
        .iter()
        .enumerate()
        .filter_map(|(i, &g)| if g.abs() > zero_tol { Some(i) } else { None })
        .collect()
}

/// Classify one candidate block's coupling to the rest of the
/// problem.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use pounce_presolve::btf::BlockTriangularBlock;
/// use pounce_presolve::coupling::{classify_block, AuxiliaryCouplingClass};
/// use pounce_presolve::incidence::{InequalityIncidence, ProbeView};
///
/// // Build a problem with one equality row (row 0) and one
/// // inequality (row 1) touching var 1, and one variable with
/// // non-zero objective gradient.
/// let p = ProbeView {
///     n_vars: 2,
///     m_rows: 2,
///     jac_irow: &[0, 0, 1],
///     jac_jcol: &[0, 1, 1],
///     jac_values: None,
///     g_l: &[0.0, 0.0],
///     g_u: &[0.0, 5.0],
///     linearity: None,
///     one_based: false,
///     eq_tol: 1e-12,
/// };
/// let ineq = InequalityIncidence::from_probe(&p);
/// let block = BlockTriangularBlock { eq_rows: vec![0], cols: vec![0, 1] };
/// let obj: HashSet<usize> = [0usize].into_iter().collect();
/// let c = classify_block(&block, &ineq, &obj);
/// assert_eq!(c, AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled);
/// ```
pub fn classify_block(
    block: &BlockTriangularBlock,
    inequalities: &InequalityIncidence,
    obj_grad_support: &HashSet<usize>,
) -> AuxiliaryCouplingClass {
    let in_obj = block.cols.iter().any(|c| obj_grad_support.contains(c));
    let in_ineq = block
        .cols
        .iter()
        .any(|&c| inequalities.var_in_inequality(c));
    match (in_obj, in_ineq) {
        (false, false) => AuxiliaryCouplingClass::PureEquality,
        (true, false) => AuxiliaryCouplingClass::ObjectiveCoupled,
        (false, true) => AuxiliaryCouplingClass::InequalityCoupled,
        (true, true) => AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incidence::ProbeView;
    use pounce_common::types::{Index, Number};

    fn probe<'a>(
        n_vars: usize,
        m_rows: usize,
        irow: &'a [Index],
        jcol: &'a [Index],
        g_l: &'a [Number],
        g_u: &'a [Number],
    ) -> ProbeView<'a> {
        ProbeView {
            n_vars,
            m_rows,
            jac_irow: irow,
            jac_jcol: jcol,
            jac_values: None,
            g_l,
            g_u,
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
        }
    }

    fn block(cols: Vec<usize>) -> BlockTriangularBlock {
        BlockTriangularBlock {
            eq_rows: cols.clone(),
            cols,
        }
    }

    #[test]
    fn coupling_pure_equality() {
        // 2 equality rows, no inequalities, zero objective gradient.
        let p = probe(2, 2, &[0, 1], &[0, 1], &[0.0, 0.0], &[0.0, 0.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let obj = HashSet::new();
        let b = block(vec![0, 1]);
        assert_eq!(
            classify_block(&b, &ineq, &obj),
            AuxiliaryCouplingClass::PureEquality
        );
    }

    #[test]
    fn coupling_objective_only() {
        let p = probe(2, 2, &[0, 1], &[0, 1], &[0.0, 0.0], &[0.0, 0.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let obj: HashSet<usize> = [0].into_iter().collect();
        let b = block(vec![0, 1]);
        assert_eq!(
            classify_block(&b, &ineq, &obj),
            AuxiliaryCouplingClass::ObjectiveCoupled
        );
    }

    #[test]
    fn coupling_inequality_only() {
        // Row 1 is g ∈ [0, 5] (inequality) and touches col 0.
        let p = probe(2, 2, &[0, 0, 1], &[0, 1, 0], &[0.0, 0.0], &[0.0, 5.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let obj = HashSet::new();
        let b = block(vec![0, 1]);
        assert_eq!(
            classify_block(&b, &ineq, &obj),
            AuxiliaryCouplingClass::InequalityCoupled
        );
    }

    #[test]
    fn coupling_both() {
        let p = probe(2, 2, &[0, 0, 1], &[0, 1, 0], &[0.0, 0.0], &[0.0, 5.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let obj: HashSet<usize> = [1].into_iter().collect();
        let b = block(vec![0, 1]);
        assert_eq!(
            classify_block(&b, &ineq, &obj),
            AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled
        );
    }

    #[test]
    fn coupling_partial_block_inequality() {
        // Inequality row touches only var 1; block has both 0 and 1.
        // Any coupling counts — block is InequalityCoupled.
        let p = probe(2, 2, &[0, 0, 1], &[0, 1, 1], &[0.0, 0.0], &[0.0, 5.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let obj = HashSet::new();
        let b = block(vec![0, 1]);
        assert_eq!(
            classify_block(&b, &ineq, &obj),
            AuxiliaryCouplingClass::InequalityCoupled
        );
    }

    #[test]
    fn coupling_zero_grad_tolerance() {
        // Variable 0 has |grad_f| = 1e-16 < zero_tol. Should not be
        // flagged as in-objective.
        let support = objective_gradient_support(&[1e-16, 0.0], 1e-12);
        assert!(support.is_empty());
        let p = probe(2, 1, &[0], &[0], &[0.0], &[0.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let b = block(vec![0, 1]);
        assert_eq!(
            classify_block(&b, &ineq, &support),
            AuxiliaryCouplingClass::PureEquality
        );
    }

    #[test]
    fn coupling_empty_block() {
        let p = probe(2, 1, &[0], &[0], &[0.0], &[0.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        let obj: HashSet<usize> = [0, 1].into_iter().collect();
        let b = BlockTriangularBlock::default();
        assert_eq!(
            classify_block(&b, &ineq, &obj),
            AuxiliaryCouplingClass::PureEquality
        );
    }

    #[test]
    fn coupling_grad_support_helper() {
        let support = objective_gradient_support(&[2.5, 0.0, -1e-3, 1e-15], 1e-9);
        let expected: HashSet<usize> = [0, 2].into_iter().collect();
        assert_eq!(support, expected);
    }
}
