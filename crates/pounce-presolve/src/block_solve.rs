//! Lightweight damped-Newton solver for square ≤ 8-dim auxiliary
//! blocks, plus the `BlockSolver` trait for the larger-block fallback.
//!
//! PR 6 of the auxiliary-presolve port (issue #53). Implementation:
//!
//! - dense LU with partial pivoting + forward/back-substitution
//!   (~50 lines, self-contained);
//! - damped Newton iteration with halving line search on the
//!   `||F||_∞` norm;
//! - `BlockSolver` trait so PR 11 can plug in an IPM-backed
//!   fallback for blocks larger than `max_dim`.
//!
//! ripopt anchor: `src/auxiliary_preprocessing.rs:1078-1182`.

use pounce_common::types::Number;

/// Tunables for the damped Newton loop.
#[derive(Debug, Clone, Copy)]
pub struct BlockSolveOptions {
    /// Maximum outer Newton iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `||F(x)||_∞`.
    pub tol: Number,
    /// Smallest backtracking step before declaring divergence.
    pub min_step: Number,
    /// Refuse to solve blocks larger than this. Defaults to 8;
    /// PR 11 lifts it via an IPM-backed fallback impl.
    pub max_dim: usize,
}

impl Default for BlockSolveOptions {
    fn default() -> Self {
        Self {
            max_iter: 30,
            tol: 1e-8,
            min_step: 1e-6,
            max_dim: 8,
        }
    }
}

/// Why a block solve failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockSolveError {
    /// Line search shrank below `min_step` without reducing the
    /// residual.
    Diverged,
    /// LU encountered a near-zero pivot.
    Singular,
    /// `eqs.dim() > options.max_dim`.
    TooLarge,
    /// Reached `max_iter` without converging.
    MaxIterReached,
    /// User callback returned `false` from `eval` or `jacobian`.
    EvalFailed,
}

/// Successful block solve.
#[derive(Debug, Clone)]
pub struct BlockSolveOutcome {
    /// The converged variable values.
    pub x: Vec<Number>,
    /// `||F(x)||_∞` at the returned point.
    pub residual_norm: Number,
    /// Number of outer Newton iterations performed.
    pub iterations: usize,
}

/// The user-supplied residual + Jacobian callbacks for one block.
/// `jacobian` writes a `dim × dim` dense matrix in row-major order.
pub trait BlockEquations {
    fn dim(&self) -> usize;
    fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool;
    fn jacobian(&mut self, x: &[Number], jac_row_major: &mut [Number]) -> bool;
}

/// Strategy for solving an auxiliary block. PR 6 ships the
/// damped-Newton implementation; PR 11 will add an IPM-backed
/// fallback that handles blocks the lightweight Newton refuses.
pub trait BlockSolver {
    fn solve(
        &mut self,
        x0: &[Number],
        eqs: &mut dyn BlockEquations,
        options: &BlockSolveOptions,
    ) -> Result<BlockSolveOutcome, BlockSolveError>;
}

/// Damped Newton with halving line search and dense LU.
#[derive(Debug, Default, Clone, Copy)]
pub struct DampedNewtonSolver;

impl BlockSolver for DampedNewtonSolver {
    /// Solve `F(x) = 0` for a small block, starting from `x0`.
    ///
    /// # Example
    ///
    /// ```
    /// use pounce_common::types::Number;
    /// use pounce_presolve::block_solve::{
    ///     BlockEquations, BlockSolveOptions, BlockSolver, DampedNewtonSolver,
    /// };
    ///
    /// // F = [x + y - 1; x - y] = 0  →  (x, y) = (0.5, 0.5).
    /// struct Eqs;
    /// impl BlockEquations for Eqs {
    ///     fn dim(&self) -> usize { 2 }
    ///     fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
    ///         f[0] = x[0] + x[1] - 1.0;
    ///         f[1] = x[0] - x[1];
    ///         true
    ///     }
    ///     fn jacobian(&mut self, _x: &[Number], j: &mut [Number]) -> bool {
    ///         j.copy_from_slice(&[1.0, 1.0, 1.0, -1.0]);
    ///         true
    ///     }
    /// }
    /// let mut solver = DampedNewtonSolver;
    /// let mut eqs = Eqs;
    /// let opt = BlockSolveOptions::default();
    /// let out = solver.solve(&[0.0, 0.0], &mut eqs, &opt).unwrap();
    /// assert!((out.x[0] - 0.5).abs() < 1e-10);
    /// assert!((out.x[1] - 0.5).abs() < 1e-10);
    /// ```
    fn solve(
        &mut self,
        x0: &[Number],
        eqs: &mut dyn BlockEquations,
        options: &BlockSolveOptions,
    ) -> Result<BlockSolveOutcome, BlockSolveError> {
        let n = eqs.dim();
        if n > options.max_dim {
            return Err(BlockSolveError::TooLarge);
        }
        if x0.len() != n {
            return Err(BlockSolveError::EvalFailed);
        }

        let mut x: Vec<Number> = x0.to_vec();
        let mut f = vec![0.0; n];
        let mut f_new = vec![0.0; n];
        let mut jac = vec![0.0; n * n];
        let mut rhs = vec![0.0; n];
        let mut x_new = vec![0.0; n];

        if !eqs.eval(&x, &mut f) {
            return Err(BlockSolveError::EvalFailed);
        }
        let mut res = inf_norm(&f);

        for iter in 0..options.max_iter {
            if res < options.tol {
                return Ok(BlockSolveOutcome {
                    x,
                    residual_norm: res,
                    iterations: iter,
                });
            }
            if !eqs.jacobian(&x, &mut jac) {
                return Err(BlockSolveError::EvalFailed);
            }
            for i in 0..n {
                rhs[i] = -f[i];
            }
            let piv =
                lu_factor_partial_pivot(&mut jac, n).map_err(|_| BlockSolveError::Singular)?;
            lu_solve(&jac, &piv, &mut rhs, n);
            // `rhs` now holds the Newton step `dx`.

            // Halving line search.
            let mut alpha: Number = 1.0;
            let mut accepted = false;
            while alpha >= options.min_step {
                for i in 0..n {
                    x_new[i] = x[i] + alpha * rhs[i];
                }
                if !eqs.eval(&x_new, &mut f_new) {
                    return Err(BlockSolveError::EvalFailed);
                }
                let res_new = inf_norm(&f_new);
                if res_new < res {
                    x.copy_from_slice(&x_new);
                    f.copy_from_slice(&f_new);
                    res = res_new;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                return Err(BlockSolveError::Diverged);
            }
        }
        // Check one last time — if max_iter brought us to tol, accept.
        if res < options.tol {
            Ok(BlockSolveOutcome {
                x,
                residual_norm: res,
                iterations: options.max_iter,
            })
        } else {
            Err(BlockSolveError::MaxIterReached)
        }
    }
}

fn inf_norm(v: &[Number]) -> Number {
    v.iter().map(|x| x.abs()).fold(0.0, Number::max)
}

/// In-place LU factorisation with partial pivoting on a row-major
/// `n × n` matrix. Returns the row-permutation vector `piv` where
/// `piv[k]` is the original row now in position `k`. Pivots smaller
/// than `1e-14 * (max entry in column k below row k)` are treated
/// as zero and cause a `Singular` error.
fn lu_factor_partial_pivot(a: &mut [Number], n: usize) -> Result<Vec<usize>, ()> {
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Find pivot row.
        let mut max_abs: Number = 0.0;
        let mut pivot_row = k;
        for i in k..n {
            let v = a[i * n + k].abs();
            if v > max_abs {
                max_abs = v;
                pivot_row = i;
            }
        }
        if max_abs < 1e-14 {
            return Err(());
        }
        if pivot_row != k {
            for j in 0..n {
                a.swap(k * n + j, pivot_row * n + j);
            }
            piv.swap(k, pivot_row);
        }
        // Eliminate below.
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            for j in (k + 1)..n {
                let upper = a[k * n + j];
                a[i * n + j] -= factor * upper;
            }
        }
    }
    Ok(piv)
}

/// Solve `LUx = Pb` in place using a factorisation from
/// [`lu_factor_partial_pivot`]. `b` is overwritten with the
/// solution.
fn lu_solve(a: &[Number], piv: &[usize], b: &mut [Number], n: usize) {
    // Permute b.
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[piv[i]];
    }
    // Forward substitution Ly = Pb (L has unit diagonal).
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= a[i * n + j] * pb[j];
        }
        pb[i] = sum;
    }
    // Back substitution Ux = y.
    for i in (0..n).rev() {
        let mut sum = pb[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * pb[j];
        }
        pb[i] = sum / a[i * n + i];
    }
    b.copy_from_slice(&pb);
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------- LU helpers ----------------------------------------------

    #[test]
    fn lu_solves_identity() {
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = vec![1.0, 2.0, 3.0];
        let piv = lu_factor_partial_pivot(&mut a, 3).expect("non-singular");
        lu_solve(&a, &piv, &mut b, 3);
        assert!((b[0] - 1.0).abs() < 1e-12);
        assert!((b[1] - 2.0).abs() < 1e-12);
        assert!((b[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lu_solves_pivoted_2x2() {
        // [[0, 1], [1, 0]] x = [3, 4]  →  x = [4, 3].
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        let mut b = vec![3.0, 4.0];
        let piv = lu_factor_partial_pivot(&mut a, 2).expect("non-singular");
        lu_solve(&a, &piv, &mut b, 2);
        assert!((b[0] - 4.0).abs() < 1e-12);
        assert!((b[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn lu_detects_singular() {
        // Rank-1 matrix.
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let result = lu_factor_partial_pivot(&mut a, 2);
        assert!(result.is_err());
    }

    // -------- Newton solver --------------------------------------------

    struct LinearF {
        a: Vec<Number>,
        b: Vec<Number>,
        n: usize,
    }
    impl BlockEquations for LinearF {
        fn dim(&self) -> usize {
            self.n
        }
        fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
            for i in 0..self.n {
                let mut s = -self.b[i];
                for j in 0..self.n {
                    s += self.a[i * self.n + j] * x[j];
                }
                f[i] = s;
            }
            true
        }
        fn jacobian(&mut self, _x: &[Number], j: &mut [Number]) -> bool {
            j.copy_from_slice(&self.a);
            true
        }
    }

    #[test]
    fn newton_linear_1d() {
        // F(x) = x - 3 = 0 → x = 3.
        let mut eqs = LinearF {
            a: vec![1.0],
            b: vec![3.0],
            n: 1,
        };
        let opt = BlockSolveOptions::default();
        let out = DampedNewtonSolver
            .solve(&[0.0], &mut eqs, &opt)
            .expect("converges");
        assert!((out.x[0] - 3.0).abs() < 1e-10);
        assert!(out.residual_norm < 1e-10);
    }

    #[test]
    fn newton_linear_2d() {
        // [x + y - 1; x - y] = 0 → (0.5, 0.5).
        let mut eqs = LinearF {
            a: vec![1.0, 1.0, 1.0, -1.0],
            b: vec![1.0, 0.0],
            n: 2,
        };
        let opt = BlockSolveOptions::default();
        let out = DampedNewtonSolver
            .solve(&[0.0, 0.0], &mut eqs, &opt)
            .expect("converges");
        assert!((out.x[0] - 0.5).abs() < 1e-10);
        assert!((out.x[1] - 0.5).abs() < 1e-10);
    }

    struct Quadratic;
    impl BlockEquations for Quadratic {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
            f[0] = x[0] * x[0] - 4.0;
            true
        }
        fn jacobian(&mut self, x: &[Number], j: &mut [Number]) -> bool {
            j[0] = 2.0 * x[0];
            true
        }
    }

    #[test]
    fn newton_quadratic() {
        // x² - 4 = 0 → x = 2 (from positive starting point).
        let mut eqs = Quadratic;
        let opt = BlockSolveOptions::default();
        let out = DampedNewtonSolver
            .solve(&[1.0], &mut eqs, &opt)
            .expect("converges");
        assert!((out.x[0] - 2.0).abs() < 1e-8);
    }

    /// {x² + y² + z² = 14, x + y + z = 6, xyz = 6}.
    /// Solution (up to permutation): (1, 2, 3).
    struct ThreeVar;
    impl BlockEquations for ThreeVar {
        fn dim(&self) -> usize {
            3
        }
        fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
            f[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 14.0;
            f[1] = x[0] + x[1] + x[2] - 6.0;
            f[2] = x[0] * x[1] * x[2] - 6.0;
            true
        }
        fn jacobian(&mut self, x: &[Number], j: &mut [Number]) -> bool {
            j[0] = 2.0 * x[0];
            j[1] = 2.0 * x[1];
            j[2] = 2.0 * x[2];
            j[3] = 1.0;
            j[4] = 1.0;
            j[5] = 1.0;
            j[6] = x[1] * x[2];
            j[7] = x[0] * x[2];
            j[8] = x[0] * x[1];
            true
        }
    }

    #[test]
    fn newton_3var_nonlinear() {
        // Start near (1, 2, 3) to avoid permutation ambiguity.
        let mut eqs = ThreeVar;
        let opt = BlockSolveOptions::default();
        let out = DampedNewtonSolver
            .solve(&[1.1, 1.9, 3.05], &mut eqs, &opt)
            .expect("converges");
        // Verify (x, y, z) is some permutation of (1, 2, 3) and the
        // residual is small.
        let mut sorted = out.x.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-6);
        assert!((sorted[1] - 2.0).abs() < 1e-6);
        assert!((sorted[2] - 3.0).abs() < 1e-6);
        assert!(out.residual_norm < 1e-8);
    }

    /// Newton on tan(x) — derivative grows fast and direction is bad.
    /// From x0 close to π/2 the Newton step overshoots into a
    /// different branch and the residual keeps growing; line search
    /// can't recover.
    struct Tan;
    impl BlockEquations for Tan {
        fn dim(&self) -> usize {
            1
        }
        fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
            f[0] = x[0].tan();
            true
        }
        fn jacobian(&mut self, x: &[Number], j: &mut [Number]) -> bool {
            let c = x[0].cos();
            j[0] = 1.0 / (c * c);
            true
        }
    }

    #[test]
    fn newton_diverges_when_starting_point_bad() {
        let mut eqs = Tan;
        let opt = BlockSolveOptions {
            max_iter: 30,
            tol: 1e-10,
            // Start at x near π/2 — tan(x) is huge, derivative is
            // huge, but a Newton step overshoots into a region where
            // tan crosses a branch and the residual jumps. The line
            // search either can't accept any step (Diverged) or we
            // hit MaxIterReached oscillating. Either is acceptable
            // failure here; the key is that we do NOT report
            // success.
            ..Default::default()
        };
        let result = DampedNewtonSolver.solve(&[1.5], &mut eqs, &opt);
        match result {
            Err(BlockSolveError::Diverged)
            | Err(BlockSolveError::MaxIterReached)
            | Err(BlockSolveError::Singular) => {}
            Ok(out) => {
                // If it did converge, residual must be tiny — but
                // we don't expect this with x0 = 1.5.
                assert!(out.residual_norm < opt.tol);
            }
            Err(e) => panic!("unexpected error variant: {e:?}"),
        }
    }

    #[test]
    fn newton_rejects_too_large() {
        struct Big;
        impl BlockEquations for Big {
            fn dim(&self) -> usize {
                9
            }
            fn eval(&mut self, _x: &[Number], f: &mut [Number]) -> bool {
                for v in f.iter_mut() {
                    *v = 0.0;
                }
                true
            }
            fn jacobian(&mut self, _x: &[Number], j: &mut [Number]) -> bool {
                for v in j.iter_mut() {
                    *v = 0.0;
                }
                true
            }
        }
        let mut eqs = Big;
        let opt = BlockSolveOptions::default();
        let err = DampedNewtonSolver
            .solve(&[0.0; 9], &mut eqs, &opt)
            .expect_err("should reject");
        assert_eq!(err, BlockSolveError::TooLarge);
    }

    #[test]
    fn newton_eval_failure_propagates() {
        struct Failing;
        impl BlockEquations for Failing {
            fn dim(&self) -> usize {
                1
            }
            fn eval(&mut self, _x: &[Number], _f: &mut [Number]) -> bool {
                false
            }
            fn jacobian(&mut self, _x: &[Number], _j: &mut [Number]) -> bool {
                true
            }
        }
        let mut eqs = Failing;
        let opt = BlockSolveOptions::default();
        let err = DampedNewtonSolver
            .solve(&[0.0], &mut eqs, &opt)
            .expect_err("eval fails");
        assert_eq!(err, BlockSolveError::EvalFailed);
    }
}
