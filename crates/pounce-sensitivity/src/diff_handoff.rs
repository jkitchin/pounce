//! The `solve → DiffHandoff` contract — the solver-agnostic bundle that
//! every differentiable solve hands to its backward pass.
//!
//! Design: `dev-notes/diff-handoff-contract.md`. The motivation is that
//! POUNCE differentiates solves across several frontends (JAX / PyTorch,
//! NLP / QP) and each was re-deriving the same *active-set* facts —
//! "a bound is active when its multiplier exceeds a tolerance", "an
//! equality row is always active", "active-bound / fixed (e.g. integer)
//! variables are pinned, `dx/dp = 0`". This struct computes those facts
//! **once**, in the producer, so every consumer reads them instead of
//! recomputing `|mult| > tol` under its own tolerance.
//!
//! This module is intentionally small and dependency-light: it is plain
//! data plus the one active-set derivation. It does *not* own the KKT
//! factor's linear algebra — that stays in [`crate::solver`] /
//! [`crate::PdSensBacksolver`]; a `DiffHandoff` produced from a live
//! solve carries the converged solution and duals, and the factor is
//! reached through the owning [`crate::Solver`] / [`ConvergedState`].
//!
//! It introduces no branch-and-bound and references no downstream
//! consumer: the test for belonging here is "would any differentiable
//! layer want it?" — and every one does.

use pounce_common::types::{Index, Number};

use crate::convenience::SensResult;

/// Default activity tolerance: a constraint or bound multiplier with
/// magnitude above this is treated as active. Matches the `_ACTIVE_TOL`
/// long used by the Python JAX/torch backward passes
/// (`python/pounce/jax/_diff.py`), centralized here so there is one
/// documented knob rather than one per frontend.
pub const DEFAULT_ACTIVE_TOL: Number = 1e-6;

/// Everything the implicit-function-theorem backward pass needs from a
/// converged solve, in a solver-agnostic shape.
///
/// Producers (IPM-NLP, convex LP/QP, conic, and — for discopt — the
/// fixed-integer leaf of a branch-and-bound) emit this; consumers
/// (`pounce.jax`, `pounce.torch`, the C ABI, a future Rust autodiff
/// user, discopt across the `solve_nlp` seam) differentiate from it.
///
/// The multiplier sign / length conventions match the existing C ABI and
/// Python `info` dict (`mult_g`, `mult_x_L`, `mult_x_U`), so this is a
/// re-shape of data POUNCE already returns — not a new computation — plus
/// the precomputed active-set masks, which are the genuinely new part.
#[derive(Debug, Clone)]
pub struct DiffHandoff {
    // ---- primal / dual solution ----
    /// Final primal iterate `x*` (length `n_x`).
    pub x: Vec<Number>,
    /// Objective value `f(x*)`.
    pub obj_val: Number,
    /// General-constraint multipliers `λ` (length `n_g`). The `g`/`G`/`A`
    /// duals, depending on the solver; one name across all of them.
    pub lambda: Vec<Number>,
    /// Variable lower-bound multipliers `z_L` (length `n_x`).
    pub mult_x_lower: Vec<Number>,
    /// Variable upper-bound multipliers `z_U` (length `n_x`).
    pub mult_x_upper: Vec<Number>,

    // ---- active set, computed ONCE here ----
    /// Constraint rows in the differentiated KKT block: equalities
    /// (always) plus inequalities whose `|λ| > active_tol`. Length `n_g`.
    /// Inactive (slack) rows drop out of the backward block.
    pub active_constraints: Vec<bool>,
    /// Variables pinned in the backward (`dx/dp = 0`): those with an
    /// active bound (`max(z_L, z_U) > active_tol`) and — for a B&B leaf —
    /// integer variables fixed at the optimum (see [`Self::pin`]).
    /// Length `n_x`.
    pub pinned_vars: Vec<bool>,
    /// The activity tolerance used to derive the masks above. Recorded so
    /// consumers and tests see the exact threshold.
    pub active_tol: Number,
}

impl DiffHandoff {
    /// Build a handoff from the raw converged solution and duals,
    /// deriving the active-set masks with `active_tol`.
    ///
    /// `equality_mask[i]` is `true` when constraint `i` is an equality
    /// (`g_l[i] == g_u[i]`) — such rows are always active. Pass an empty
    /// slice when there are no general constraints.
    pub fn from_solution(
        x: Vec<Number>,
        obj_val: Number,
        lambda: Vec<Number>,
        mult_x_lower: Vec<Number>,
        mult_x_upper: Vec<Number>,
        equality_mask: &[bool],
        active_tol: Number,
    ) -> Self {
        let n_x = x.len();
        let n_g = lambda.len();
        debug_assert_eq!(mult_x_lower.len(), n_x, "z_L length must match x");
        debug_assert_eq!(mult_x_upper.len(), n_x, "z_U length must match x");
        debug_assert!(
            equality_mask.is_empty() || equality_mask.len() == n_g,
            "equality_mask must be empty or length n_g"
        );

        // A bound is active when either side's multiplier exceeds the
        // tolerance → the variable is pinned (dx/dp = 0).
        let pinned_vars: Vec<bool> = (0..n_x)
            .map(|i| mult_x_lower[i] > active_tol || mult_x_upper[i] > active_tol)
            .collect();

        // A constraint row is active when it is an equality (always) or
        // its multiplier magnitude exceeds the tolerance.
        let active_constraints: Vec<bool> = (0..n_g)
            .map(|i| {
                let is_eq = equality_mask.get(i).copied().unwrap_or(false);
                is_eq || lambda[i].abs() > active_tol
            })
            .collect();

        Self {
            x,
            obj_val,
            lambda,
            mult_x_lower,
            mult_x_upper,
            active_constraints,
            pinned_vars,
            active_tol,
        }
    }

    /// Re-shape a [`SensResult`] from a converged solve into a
    /// `DiffHandoff`, using [`DEFAULT_ACTIVE_TOL`].
    ///
    /// Returns `None` when the solve did not populate the duals
    /// (`mult_g` / `mult_x_l` / `mult_x_u`) — i.e. it didn't converge, or
    /// the NLP didn't expose user-space multipliers. `equality_mask` is
    /// the caller's `g_l[i] == g_u[i]` test (length `n_g`); pass an empty
    /// slice if unknown, in which case only nonzero-multiplier rows count
    /// as active (a conservative, still-correct choice for equalities
    /// whose multipliers are nonzero at the solution).
    pub fn from_sens_result(res: &SensResult, equality_mask: &[bool]) -> Option<Self> {
        let x = res.x.clone()?;
        let obj_val = res.obj_val?;
        let lambda = res.mult_g.clone()?;
        let mult_x_lower = res.mult_x_l.clone()?;
        let mult_x_upper = res.mult_x_u.clone()?;
        Some(Self::from_solution(
            x,
            obj_val,
            lambda,
            mult_x_lower,
            mult_x_upper,
            equality_mask,
            DEFAULT_ACTIVE_TOL,
        ))
    }

    /// Additionally pin a set of variables — the seam discopt uses for a
    /// branch-and-bound leaf: integer variables fixed at the optimum
    /// differentiate exactly like active bounds (`dx/dp = 0`). Indices
    /// out of range are ignored.
    pub fn pin(&mut self, indices: &[Index]) {
        for &i in indices {
            if i < 0 {
                continue;
            }
            if let Some(slot) = self.pinned_vars.get_mut(i as usize) {
                *slot = true;
            }
        }
    }

    /// Number of primal variables.
    pub fn n_x(&self) -> usize {
        self.x.len()
    }

    /// Number of general constraints.
    pub fn n_g(&self) -> usize {
        self.lambda.len()
    }

    /// Count of pinned variables (active bounds + any [`Self::pin`]ned).
    pub fn n_pinned(&self) -> usize {
        self.pinned_vars.iter().filter(|&&b| b).count()
    }

    /// Count of active constraint rows.
    pub fn n_active_constraints(&self) -> usize {
        self.active_constraints.iter().filter(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_active_bounds_and_marks_active_constraints() {
        // x0: lower bound active (z_L large). x1: free. x2: upper active.
        let x = vec![0.0, 1.0, 2.0];
        let z_l = vec![5.0, 0.0, 0.0];
        let z_u = vec![0.0, 0.0, 3.0];
        // g0: equality. g1: inactive inequality (λ≈0). g2: active inequality.
        let lambda = vec![0.0, 1e-9, 4.0];
        let eq = vec![true, false, false];

        let h = DiffHandoff::from_solution(
            x, 42.0, lambda, z_l, z_u, &eq, DEFAULT_ACTIVE_TOL,
        );

        assert_eq!(h.pinned_vars, vec![true, false, true]);
        assert_eq!(h.active_constraints, vec![true, false, true]);
        assert_eq!(h.n_pinned(), 2);
        assert_eq!(h.n_active_constraints(), 2);
        assert_eq!(h.obj_val, 42.0);
    }

    #[test]
    fn empty_equality_mask_treats_only_nonzero_rows_as_active() {
        let h = DiffHandoff::from_solution(
            vec![0.0],
            0.0,
            vec![0.0, 5.0],
            vec![0.0],
            vec![0.0],
            &[],
            DEFAULT_ACTIVE_TOL,
        );
        assert_eq!(h.active_constraints, vec![false, true]);
    }

    #[test]
    fn pin_adds_integer_variables() {
        let mut h = DiffHandoff::from_solution(
            vec![0.0, 0.0, 0.0],
            0.0,
            vec![],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            &[],
            DEFAULT_ACTIVE_TOL,
        );
        assert_eq!(h.n_pinned(), 0);
        h.pin(&[1, 99]); // 99 is out of range, ignored
        assert_eq!(h.pinned_vars, vec![false, true, false]);
        assert_eq!(h.n_pinned(), 1);
    }
}
