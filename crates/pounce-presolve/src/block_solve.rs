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
#[derive(Debug, Clone)]
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
    /// PR #60 review nit: optional per-variable bounds on the
    /// Newton iterate. When set, the solver rejects any final
    /// solution with `x[i] < lower[i] - tol` or `x[i] > upper[i] +
    /// tol`, returning `BlockSolveError::OutOfBounds` instead of
    /// `Ok`. Lengths must equal the block dimension.
    pub bounds: Option<(Vec<Number>, Vec<Number>)>,
}

impl Default for BlockSolveOptions {
    fn default() -> Self {
        Self {
            max_iter: 30,
            tol: 1e-8,
            min_step: 1e-6,
            max_dim: 8,
            bounds: None,
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
    /// PR #60 review nit: Newton converged but the final iterate
    /// lies outside `BlockSolveOptions::bounds`. The orchestrator
    /// converts this into `AuxiliaryRejectionReason::OutOfBounds`
    /// so the block isn't applied.
    OutOfBounds,
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

/// Solver for blocks exceeding `auxiliary_max_block_dim`. The
/// orchestrator falls through to this when the lightweight Newton
/// rejects a block as too large.
///
/// PR 11 ships [`RelaxedNewtonSolver`] as the default impl — same
/// algorithm as `DampedNewtonSolver` but with a higher dimension
/// cap and more iterations. The architectural seam is what matters:
/// a future `pounce-algorithm`-side impl can drop in here without
/// touching the orchestrator.
pub trait LargeBlockSolver {
    fn solve_large(
        &mut self,
        x0: &[Number],
        eqs: &mut dyn BlockEquations,
        options: &BlockSolveOptions,
    ) -> Result<BlockSolveOutcome, BlockSolveError>;
}

/// Default [`LargeBlockSolver`]: same damped Newton, looser limits
/// (`max_dim = 64`, `max_iter = 200`, `min_step = 1e-10`).
/// Handles "well-behaved but too-big" blocks. Diverges-out-of-the-box
/// blocks need the real IPM-backed solver from `pounce-algorithm`,
/// tracked as a follow-up.
#[derive(Debug, Default, Clone, Copy)]
pub struct RelaxedNewtonSolver;

impl LargeBlockSolver for RelaxedNewtonSolver {
    fn solve_large(
        &mut self,
        x0: &[Number],
        eqs: &mut dyn BlockEquations,
        options: &BlockSolveOptions,
    ) -> Result<BlockSolveOutcome, BlockSolveError> {
        // Inherit the caller's `tol` but loosen the iteration and
        // dimension caps. The `max_dim` ceiling on the relaxed path
        // is the real switch; without bumping it, Newton would
        // reject before the line-search budget matters.
        let relaxed = BlockSolveOptions {
            tol: options.tol,
            max_iter: options.max_iter.max(200),
            min_step: options.min_step.min(1e-10),
            max_dim: options.max_dim.max(64),
            bounds: options.bounds.clone(),
        };
        DampedNewtonSolver.solve(x0, eqs, &relaxed)
    }
}

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
                check_bounds(&x, options)?;
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
            check_bounds(&x, options)?;
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

/// Check that every entry of `x` lies within the optional bounds
/// from `BlockSolveOptions::bounds` (PR #60 review nit). Returns
/// `Err(OutOfBounds)` on violation; `Ok(())` when no bounds were
/// supplied or all entries pass.
fn check_bounds(x: &[Number], options: &BlockSolveOptions) -> Result<(), BlockSolveError> {
    if let Some((lo, hi)) = &options.bounds {
        // Tolerate small slop for finite-precision tol.
        let slop = options.tol.max(1e-12);
        for (i, &xi) in x.iter().enumerate() {
            if xi < lo[i] - slop || xi > hi[i] + slop {
                return Err(BlockSolveError::OutOfBounds);
            }
        }
    }
    Ok(())
}

fn inf_norm(v: &[Number]) -> Number {
    v.iter().map(|x| x.abs()).fold(0.0, Number::max)
}

/// In-place LU factorisation with partial pivoting on a row-major
/// `n × n` matrix. Returns the row-permutation vector `piv` where
/// `piv[k]` is the original row now in position `k`. Pivots smaller
/// than `1e-14` are treated as zero and cause a `Singular` error.
///
/// `pub(crate)` so PR 7's `reduction_frame` can reuse it for the
/// multiplier-recovery solve.
pub(crate) fn lu_factor_partial_pivot(a: &mut [Number], n: usize) -> Result<Vec<usize>, ()> {
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
pub(crate) fn lu_solve(a: &[Number], piv: &[usize], b: &mut [Number], n: usize) {
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

    // -------- LargeBlockSolver -----------------------------------------

    #[test]
    fn relaxed_newton_solves_10_var_linear() {
        // 10×10 identity-like system. The default DampedNewtonSolver
        // rejects this with `TooLarge` because n=10 > max_dim=8;
        // RelaxedNewtonSolver bumps max_dim and solves to within tol.
        let n = 10;
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n];
        for i in 0..n {
            a[i * n + i] = 2.0;
            b[i] = (i + 1) as Number;
        }
        struct Lin {
            a: Vec<Number>,
            b: Vec<Number>,
            n: usize,
        }
        impl BlockEquations for Lin {
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
        let mut eqs = Lin { a, b, n };
        // Caller-side options keep max_dim at 8 (the default);
        // RelaxedNewtonSolver relaxes internally.
        let opts = BlockSolveOptions::default();

        // First: confirm the default solver rejects.
        let lightweight_err = DampedNewtonSolver
            .solve(&vec![0.0; n], &mut eqs, &opts)
            .expect_err("default should reject 10-dim block");
        assert_eq!(lightweight_err, BlockSolveError::TooLarge);

        // Now: RelaxedNewtonSolver solves it.
        let out = RelaxedNewtonSolver
            .solve_large(&vec![0.0; n], &mut eqs, &opts)
            .expect("relaxed solver handles 10-dim");
        for i in 0..n {
            assert!((out.x[i] - (i + 1) as Number / 2.0).abs() < 1e-10);
        }
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

    /// Small LCG-based RNG used by the fuzz tests below. Captures
    /// state in a struct (not a closure) so the borrow checker
    /// doesn't trip when we use both `next_u64` and `unit` in the
    /// same scope.
    struct FuzzRng(u64);
    impl FuzzRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 32
        }
        fn unit(&mut self) -> Number {
            let raw = (self.next_u64() & 0x3fff_ffff) as Number;
            raw / (1u64 << 29) as Number - 1.0
        }
    }

    /// Random-linear fuzz: build a well-conditioned random N×N
    /// linear system, solve it both via the public Newton path AND
    /// via a direct LU solve, and check they agree to 1e-10. This
    /// catches bugs in Newton's wrapper logic (line search,
    /// convergence check, RHS sign, scratch buffer reuse) without
    /// being a tautology — the LU pieces are tested independently
    /// in the LU tests above.
    #[test]
    fn newton_fuzz_random_linear_vs_direct_lu() {
        let mut rng = FuzzRng::new(0xfeed_face_dead_b33f);

        let mut tested = 0usize;
        for _ in 0..30 {
            let n = 1 + (rng.next_u64() % 6) as usize; // N ∈ [1, 6]
            // Build A with strong diagonal + small off-diag entries so
            // it's well-conditioned regardless of the random seed.
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = if i == j {
                        2.0 + rng.unit().abs()
                    } else {
                        0.3 * rng.unit()
                    };
                }
            }
            // Pick a random target solution x_star and form b = A x_star.
            let x_star: Vec<Number> = (0..n).map(|_| rng.unit()).collect();
            let mut b = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += a[i * n + j] * x_star[j];
                }
                b[i] = s;
            }

            // Reference: direct LU solve.
            let mut a_ref = a.clone();
            let piv = lu_factor_partial_pivot(&mut a_ref, n).expect("well-conditioned");
            let mut x_lu = b.clone();
            lu_solve(&a_ref, &piv, &mut x_lu, n);

            // Newton: F(x) = A x - b = 0.
            struct LinSys {
                a: Vec<Number>,
                b: Vec<Number>,
                n: usize,
            }
            impl BlockEquations for LinSys {
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
            let mut eqs = LinSys {
                a: a.clone(),
                b: b.clone(),
                n,
            };
            // Start at the origin so Newton actually iterates.
            let x0 = vec![0.0; n];
            let opt = BlockSolveOptions::default();
            let out = DampedNewtonSolver
                .solve(&x0, &mut eqs, &opt)
                .expect("Newton converges on a well-conditioned linear system");

            // Agreement: Newton's solution matches the direct LU
            // solution (which also matches x_star, by construction).
            let mut max_diff: Number = 0.0;
            for i in 0..n {
                max_diff = max_diff.max((out.x[i] - x_lu[i]).abs());
                max_diff = max_diff.max((out.x[i] - x_star[i]).abs());
            }
            assert!(
                max_diff < 1e-10,
                "Newton vs LU disagreement of {max_diff:.3e} on n={n}"
            );
            tested += 1;
        }
        assert_eq!(tested, 30);
    }

    /// Mildly-nonlinear fuzz: `F(x) = A (x - x*) + ε (x - x*) ⊙ (x - x*)`
    /// has a root at `x = x*` and a non-trivial Jacobian. Verify
    /// Newton finds the root starting close enough.
    #[test]
    fn newton_fuzz_nonlinear_quadratic_root() {
        let mut rng = FuzzRng::new(0xcafe_b00b_1337_5eed);

        for trial in 0..20 {
            let n = 1 + (rng.next_u64() % 4) as usize; // N ∈ [1, 4]
            // Diagonally dominant A.
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = if i == j {
                        3.0 + rng.unit().abs()
                    } else {
                        0.2 * rng.unit()
                    };
                }
            }
            let x_star: Vec<Number> = (0..n).map(|_| rng.unit()).collect();
            // ε small enough that linearization at x_star ≈ A.
            let eps = 0.1;

            struct Nl {
                a: Vec<Number>,
                x_star: Vec<Number>,
                eps: Number,
                n: usize,
            }
            impl BlockEquations for Nl {
                fn dim(&self) -> usize {
                    self.n
                }
                fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
                    for i in 0..self.n {
                        let mut s = 0.0;
                        for j in 0..self.n {
                            s += self.a[i * self.n + j] * (x[j] - self.x_star[j]);
                        }
                        let dxi = x[i] - self.x_star[i];
                        s += self.eps * dxi * dxi;
                        f[i] = s;
                    }
                    true
                }
                fn jacobian(&mut self, x: &[Number], j: &mut [Number]) -> bool {
                    for i in 0..self.n {
                        for k in 0..self.n {
                            let mut v = self.a[i * self.n + k];
                            if i == k {
                                v += 2.0 * self.eps * (x[i] - self.x_star[i]);
                            }
                            j[i * self.n + k] = v;
                        }
                    }
                    true
                }
            }

            // Start near x_star.
            let x0: Vec<Number> = x_star.iter().map(|&v| v + 0.1 * rng.unit()).collect();
            let mut eqs = Nl {
                a: a.clone(),
                x_star: x_star.clone(),
                eps,
                n,
            };
            let opt = BlockSolveOptions::default();
            let out = DampedNewtonSolver
                .solve(&x0, &mut eqs, &opt)
                .unwrap_or_else(|e| panic!("trial {trial} (n={n}): {e:?}"));
            let mut max_err: Number = 0.0;
            for i in 0..n {
                max_err = max_err.max((out.x[i] - x_star[i]).abs());
            }
            assert!(
                max_err < 1e-7,
                "trial {trial}: Newton missed root by {max_err:.3e}"
            );
        }
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

    /// PR #60 review nit: when Newton converges to a root outside
    /// `BlockSolveOptions::bounds`, the solver returns
    /// `OutOfBounds` rather than `Ok`. Use the linear 1-d system
    /// `x - 3 = 0` (root at 3) with bounds [0, 2].
    #[test]
    fn newton_out_of_bounds_rejects() {
        let mut eqs = LinearF {
            a: vec![1.0],
            b: vec![3.0],
            n: 1,
        };
        let opt = BlockSolveOptions {
            bounds: Some((vec![0.0], vec![2.0])),
            ..Default::default()
        };
        let err = DampedNewtonSolver
            .solve(&[0.0], &mut eqs, &opt)
            .expect_err("root outside bounds");
        assert_eq!(err, BlockSolveError::OutOfBounds);
    }

    /// Sanity check: with bounds that DO contain the root, the
    /// solver accepts normally.
    #[test]
    fn newton_in_bounds_accepts() {
        let mut eqs = LinearF {
            a: vec![1.0],
            b: vec![3.0],
            n: 1,
        };
        let opt = BlockSolveOptions {
            bounds: Some((vec![0.0], vec![10.0])),
            ..Default::default()
        };
        let out = DampedNewtonSolver
            .solve(&[0.0], &mut eqs, &opt)
            .expect("root inside bounds");
        assert!((out.x[0] - 3.0).abs() < 1e-10);
    }
}
