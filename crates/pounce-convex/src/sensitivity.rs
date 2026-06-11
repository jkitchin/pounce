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
//! default `1e-8`) is placed on the diagonal so the indefinite factor is
//! stable; the induced error in the step is `O(δ)`.

use crate::ipm::QpOptions;
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_common::types::{Index, Number};
use pounce_linalg::symmetric_eigen;
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// A reason a [`QpSensitivity`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensError {
    /// The solution was not optimal, so the active set is undefined.
    NotOptimal,
    /// The active-set KKT factorization failed (e.g. the active constraint
    /// gradients are rank-deficient, so the parametric step is not unique).
    FactorizationFailed,
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
    fact: Factorization,
    /// Problem data, retained for the reduced-Hessian projection.
    prob: QpProblem,
    /// Active inequality rows (indices into `G`).
    active_ineq: Vec<usize>,
    /// Variables whose bound is active (one `eⱼᵀ` row each).
    active_bound_vars: Vec<usize>,
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
    /// Returns [`SensError::NotOptimal`] if `sol` is not optimal, or
    /// [`SensError::FactorizationFailed`] if the active-set KKT is singular.
    pub fn build<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        mut make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        if sol.status != QpStatus::Optimal {
            return Err(SensError::NotOptimal);
        }
        let n = prob.n;
        let m_eq = prob.m_eq();
        let reg = opts.reg;

        // Active set: which inequality rows and which variable bounds bind.
        let active_ineq: Vec<usize> = (0..prob.m_ineq())
            .filter(|&i| sol.z[i] > active_tol)
            .collect();
        // A bound contributes one row `eⱼᵀ` (the gradient of `xⱼ = const` is
        // `eⱼ` whether the lower or upper bound is the active one).
        let active_bound_vars: Vec<usize> = (0..n)
            .filter(|&j| sol.z_lb[j] > active_tol || sol.z_ub[j] > active_tol)
            .collect();
        let n_active = active_ineq.len() + active_bound_vars.len();
        let dim = n + m_eq + n_active;

        // Assemble the lower triangle of the symmetric KKT matrix.
        let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut add = |r: usize, c: usize, v: f64| {
            let (r, c) = if r >= c { (r, c) } else { (c, r) };
            *entries.entry((r, c)).or_insert(0.0) += v;
        };

        // (x,x): P + δI.
        for t in &prob.p_lower {
            add(t.row, t.col, t.val);
        }
        for i in 0..n {
            add(i, i, reg);
        }
        // (y,x): A; (y,y): −δI.
        for t in &prob.a {
            add(n + t.row, t.col, t.val);
        }
        for i in 0..m_eq {
            add(n + i, n + i, -reg);
        }
        // Active-row block `B_a` after the equality rows, in order:
        // active inequality rows, then active bound rows. (·,·): −δI diagonal.
        let abase = n + m_eq;
        for (k, &i) in active_ineq.iter().enumerate() {
            // The k-th active row holds G's row i.
            for t in prob.g.iter().filter(|t| t.row == i) {
                add(abase + k, t.col, t.val);
            }
        }
        for (k, &j) in active_bound_vars.iter().enumerate() {
            add(abase + active_ineq.len() + k, j, 1.0);
        }
        for k in 0..n_active {
            add(abase + k, abase + k, -reg);
        }

        // Triplets → 1-based lower-triangle arrays for the factorization.
        let nnz = entries.len();
        let mut airn = Vec::with_capacity(nnz);
        let mut ajcn = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for ((r, c), v) in entries {
            airn.push((r + 1) as Index);
            ajcn.push((c + 1) as Index);
            values.push(v);
        }

        let fact = Factorization::new(dim as Index, airn, ajcn, values, make_backend())
            .map_err(|_| SensError::FactorizationFailed)?;

        Ok(QpSensitivity {
            n,
            m_eq,
            dim,
            fact,
            prob: prob.clone(),
            active_ineq,
            active_bound_vars,
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
    pub fn step_from_db(&mut self, db: &[f64]) -> Vec<f64> {
        assert_eq!(db.len(), self.m_eq, "db must have length m_eq");
        let mut rhs = vec![0.0 as Number; self.dim];
        rhs[self.n..self.n + self.m_eq].copy_from_slice(db);
        // A singular factor would have been caught at build; a back-solve
        // failure here is not recoverable, so surface a zero step.
        if self.fact.solve_one(&mut rhs).is_err() {
            return vec![0.0; self.n];
        }
        rhs.truncate(self.n);
        rhs
    }

    /// The active-set KKT dimension `n + m_eq + n_active`.
    pub fn kkt_dim(&self) -> usize {
        self.dim
    }

    /// Reduced Hessian of the QP at the optimum: the objective Hessian `P`
    /// projected onto the null space of the **active constraints**
    /// `B = [A; active G rows; active bound rows]`. If `Z` is an
    /// orthonormal basis of `null(B)` (the feasible directions / degrees of
    /// freedom), the reduced Hessian is `H_R = Zᵀ P Z`. Its eigenvalues are
    /// the objective's curvatures along feasible directions: all positive
    /// ⟺ a strict second-order minimizer (always so for a strictly convex
    /// `P`), and their spread is the conditioning of the QP on the active
    /// manifold. This mirrors the NLP `Solver.reduced_hessian` /
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
    pub fn reduced_hessian(&self, rank_tol: f64) -> ReducedHessian {
        let n = self.n;

        // Active Jacobian B (m_act × n), dense row-major: equality rows,
        // then active inequality rows, then active variable-bound rows.
        let m_act = self.m_eq + self.active_ineq.len() + self.active_bound_vars.len();
        let mut b = vec![0.0; m_act * n];
        for t in &self.prob.a {
            b[t.row * n + t.col] += t.val;
        }
        let mut row = self.m_eq;
        for &i in &self.active_ineq {
            for t in self.prob.g.iter().filter(|t| t.row == i) {
                b[row * n + t.col] += t.val;
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
        symmetric_eigen(&btb, n, &mut sv, &mut vecs); // ascending eigenvalues

        // rank(B) = # squared-singular-values above the relative threshold;
        // the null space is spanned by the eigenvectors of the rest (the
        // smallest, ≈ 0). With ascending order those are the first columns.
        let max_sv = sv.last().copied().unwrap_or(0.0).max(0.0);
        let thresh = rank_tol * max_sv;
        let rank = sv.iter().filter(|&&l| l > thresh).count();
        let n_dof = n - rank;

        // Dense symmetric P (n×n) from its lower triangle.
        let mut p = vec![0.0; n * n];
        for t in &self.prob.p_lower {
            p[t.row * n + t.col] += t.val;
            if t.row != t.col {
                p[t.col * n + t.row] += t.val;
            }
        }

        // H_R = Zᵀ P Z, with Z = first `n_dof` columns of `vecs` (the null
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
        symmetric_eigen(&hr, n_dof, &mut eigenvalues, &mut eigenvectors);

        ReducedHessian {
            n_dof,
            matrix: hr,
            eigenvalues,
            eigenvectors,
        }
    }

    /// [`reduced_hessian`](Self::reduced_hessian) with a relative rank
    /// tolerance of `1e-9`.
    pub fn reduced_hessian_default(&self) -> ReducedHessian {
        self.reduced_hessian(1e-9)
    }
}

/// The reduced Hessian `H_R = Zᵀ P Z` of a QP on its active manifold, with
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
    /// `eigenvalues[j]`.
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
        let rh = sens.reduced_hessian_default();
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
        let rh = sens.reduced_hessian_default();
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
        let rh = sens.reduced_hessian_default();
        assert_eq!(rh.n_dof, 1);
        assert!(
            (rh.eigenvalues[0] - 1.5).abs() < 1e-9,
            "H_R = {:?}",
            rh.eigenvalues
        );
        assert!((rh.matrix[0] - 1.5).abs() < 1e-9);
    }
}
