//! Dense general matrix — port of `LinAlg/IpDenseGenMatrix.{hpp,cpp}`.
//!
//! Storage is column-major Fortran order: `values[i + j * n_rows]`
//! references row `i`, column `j`. Phase 2 ships the BLAS-2 paths
//! (`mult_vector`, `trans_mult_vector`, `Copy`, `FillIdentity`,
//! `ScaleColumns`, row/col amax, has_valid_numbers). The Cholesky
//! factor + back-solves (`compute_cholesky_factor`,
//! `cholesky_back_solve_matrix`, `cholesky_solve_vector/matrix`) and
//! `high_rank_update_transpose` (used by the L-BFGS aug-system solver)
//! are implemented as pure-Rust column-major routines. The remaining
//! LAPACK/BLAS-fronted upstream routines — eigenvectors (`dsyev`), LU
//! factor + solve (`dgetrf`/`dgetrs`), and the dense `gemm`
//! (`add_matrix_product`) — have no caller in pounce and were never
//! ported; they are intentionally omitted.
//!
//! `mult_vector_impl` / `trans_mult_vector_impl` use the reference
//! DGEMV scalar-loop order (column-major outer over `j`) so the
//! floating-point accumulation is deterministic and BLAS-implementation
//! independent. The BLAS-linked path can replace this once the LAPACK
//! shim is wired.

use crate::dense_vector::DenseVector;
use crate::matrix::{Matrix, MatrixCache};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug)]
pub struct DenseGenMatrixSpace {
    n_rows: Index,
    n_cols: Index,
}

impl DenseGenMatrixSpace {
    pub fn new(n_rows: Index, n_cols: Index) -> Rc<Self> {
        Rc::new(Self { n_rows, n_cols })
    }

    pub fn n_rows(&self) -> Index {
        self.n_rows
    }
    pub fn n_cols(&self) -> Index {
        self.n_cols
    }

    pub fn make_new_dense_gen(self: &Rc<Self>) -> DenseGenMatrix {
        DenseGenMatrix::new(Rc::clone(self))
    }
}

#[derive(Debug)]
pub struct DenseGenMatrix {
    space: Rc<DenseGenMatrixSpace>,
    cache: MatrixCache,
    values: Vec<Number>,
    initialized: Cell<bool>,
}

impl DenseGenMatrix {
    pub fn new(space: Rc<DenseGenMatrixSpace>) -> Self {
        let n = (space.n_rows.max(0) as usize) * (space.n_cols.max(0) as usize);
        Self {
            space,
            cache: MatrixCache::new(),
            values: vec![0.0; n],
            initialized: Cell::new(false),
        }
    }

    pub fn space(&self) -> &Rc<DenseGenMatrixSpace> {
        &self.space
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.get()
    }

    /// Read-only column-major slice. Panics in debug if not initialized.
    pub fn values(&self) -> &[Number] {
        debug_assert!(self.initialized.get());
        &self.values
    }

    /// Mutable column-major slice. Marks the matrix initialized and
    /// bumps the change tag (mirrors upstream's non-const `Values()`).
    pub fn values_mut(&mut self) -> &mut [Number] {
        self.initialized.set(true);
        self.cache.bump();
        &mut self.values
    }

    fn nr(&self) -> usize {
        self.space.n_rows.max(0) as usize
    }
    fn nc(&self) -> usize {
        self.space.n_cols.max(0) as usize
    }

    pub fn copy_from(&mut self, m: &DenseGenMatrix) {
        debug_assert_eq!(self.space.n_rows, m.space.n_rows);
        debug_assert_eq!(self.space.n_cols, m.space.n_cols);
        self.values.copy_from_slice(m.values());
        self.initialized.set(true);
        self.cache.bump();
    }

    /// `M ← factor · I`. Requires square.
    pub fn fill_identity(&mut self, factor: Number) {
        debug_assert_eq!(self.space.n_rows, self.space.n_cols);
        let n = self.nr();
        self.values.iter_mut().for_each(|v| *v = 0.0);
        if factor != 0.0 {
            for i in 0..n {
                self.values[i + i * n] = factor;
            }
        }
        self.initialized.set(true);
        self.cache.bump();
    }

    /// Scale column `j` by `scal_vec[j]` for all `j`.
    pub fn scale_columns(&mut self, scal_vec: &DenseVector) {
        debug_assert_eq!(scal_vec.dim(), self.space.n_cols);
        debug_assert!(self.initialized.get());
        let nr = self.nr();
        let nc = self.nc();
        let scals = scal_vec.expanded_values();
        for (j, &s) in scals.iter().enumerate().take(nc) {
            let col = &mut self.values[j * nr..j * nr + nr];
            for v in col.iter_mut() {
                *v *= s;
            }
        }
        self.cache.bump();
    }

    /// `J Jᵀ = M` where `M` is symmetric positive definite. Lower
    /// triangle of `J` is written column-major; strict-upper is zeroed
    /// to match upstream's post-`dpotrf` cleanup
    /// (`IpDenseGenMatrix.cpp:185-191`). Returns `false` if `M` is not
    /// PD (matches LAPACK's `info != 0` path). Pure-Rust column-major
    /// Cholesky — for the L-BFGS aug-system solver where `n` is the
    /// memory window (≤ 20) so a hand-coded variant is plenty fast and
    /// keeps the linalg crate dependency-free until the LAPACK shim
    /// lands.
    pub fn compute_cholesky_factor(&mut self, m: &crate::dense_sym_matrix::DenseSymMatrix) -> bool {
        let dim = m.n_rows() as usize;
        debug_assert_eq!(dim, self.nr());
        debug_assert_eq!(dim, self.nc());
        // Copy the lower triangle of M into self.
        let mvals = m.values();
        for j in 0..dim {
            for i in j..dim {
                self.values[i + j * dim] = mvals[i + j * dim];
            }
        }
        // Column-major right-looking Cholesky (matches dpotrf 'L').
        for j in 0..dim {
            // Diagonal: L[j,j] = sqrt(M[j,j] - Σ_{k<j} L[j,k]²)
            let mut diag = self.values[j + j * dim];
            for k in 0..j {
                let ljk = self.values[j + k * dim];
                diag -= ljk * ljk;
            }
            if diag <= 0.0 || !diag.is_finite() {
                self.initialized.set(false);
                self.cache.bump();
                return false;
            }
            let ljj = diag.sqrt();
            self.values[j + j * dim] = ljj;
            // Below-diagonal: L[i,j] = (M[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
            for i in (j + 1)..dim {
                let mut s = self.values[i + j * dim];
                for k in 0..j {
                    s -= self.values[i + k * dim] * self.values[j + k * dim];
                }
                self.values[i + j * dim] = s / ljj;
            }
        }
        // Zero the strict upper triangle.
        for j in 1..dim {
            for i in 0..j {
                self.values[i + j * dim] = 0.0;
            }
        }
        self.initialized.set(true);
        self.cache.bump();
        true
    }

    /// `B ← α · op(L)⁻¹ · B` where `L` is the lower-triangular factor
    /// stored in `self`. `trans = true` solves `Lᵀ X = α B`,
    /// `trans = false` solves `L X = α B`. Mirrors `dtrsm` for a single
    /// triangular factor in-place per upstream `IpDenseGenMatrix.cpp:228-240`.
    pub fn cholesky_back_solve_matrix(&self, trans: bool, alpha: Number, b: &mut DenseGenMatrix) {
        debug_assert!(self.initialized.get());
        debug_assert_eq!(self.nr(), self.nc());
        debug_assert_eq!(self.nr(), b.nr());
        let dim = self.nr();
        let nrhs = b.nc();
        let lvals = self.values.clone();
        let bvals = b.values_mut();
        if alpha != 1.0 {
            for v in bvals.iter_mut() {
                *v *= alpha;
            }
        }
        for col in 0..nrhs {
            let base = col * dim;
            if !trans {
                // Forward solve L · x = b.
                for i in 0..dim {
                    let mut s = bvals[base + i];
                    for k in 0..i {
                        s -= lvals[i + k * dim] * bvals[base + k];
                    }
                    bvals[base + i] = s / lvals[i + i * dim];
                }
            } else {
                // Back solve Lᵀ · x = b.
                for i in (0..dim).rev() {
                    let mut s = bvals[base + i];
                    for k in (i + 1)..dim {
                        s -= lvals[k + i * dim] * bvals[base + k];
                    }
                    bvals[base + i] = s / lvals[i + i * dim];
                }
            }
        }
    }

    /// Solve `(L Lᵀ) x = b` in place. Mirrors LAPACK `dpotrs` for a
    /// single right-hand side.
    pub fn cholesky_solve_vector(&self, b: &mut DenseVector) {
        debug_assert!(self.initialized.get());
        debug_assert_eq!(self.nr(), self.nc());
        debug_assert_eq!(self.nr() as usize, b.dim() as usize);
        let dim = self.nr();
        let lvals = &self.values;
        let bv = b.values_mut();
        // L · y = b
        for i in 0..dim {
            let mut s = bv[i];
            for k in 0..i {
                s -= lvals[i + k * dim] * bv[k];
            }
            bv[i] = s / lvals[i + i * dim];
        }
        // Lᵀ · x = y
        for i in (0..dim).rev() {
            let mut s = bv[i];
            for k in (i + 1)..dim {
                s -= lvals[k + i * dim] * bv[k];
            }
            bv[i] = s / lvals[i + i * dim];
        }
    }

    /// Solve `(L Lᵀ) X = B` in place, column-by-column.
    pub fn cholesky_solve_matrix(&self, b: &mut DenseGenMatrix) {
        debug_assert!(self.initialized.get());
        debug_assert_eq!(self.nr(), self.nc());
        debug_assert_eq!(self.nr(), b.nr());
        let dim = self.nr();
        let nrhs = b.nc();
        let lvals = self.values.clone();
        let bvals = b.values_mut();
        for col in 0..nrhs {
            let base = col * dim;
            for i in 0..dim {
                let mut s = bvals[base + i];
                for k in 0..i {
                    s -= lvals[i + k * dim] * bvals[base + k];
                }
                bvals[base + i] = s / lvals[i + i * dim];
            }
            for i in (0..dim).rev() {
                let mut s = bvals[base + i];
                for k in (i + 1)..dim {
                    s -= lvals[k + i * dim] * bvals[base + k];
                }
                bvals[base + i] = s / lvals[i + i * dim];
            }
        }
    }

    /// `M[i, j] ← α · V1[:, i]ᵀ · V2[:, j] + β · M[i, j]` for the full
    /// rectangle. Port of `DenseGenMatrix::HighRankUpdateTranspose`
    /// (`IpDenseGenMatrix.cpp:117-150`).
    pub fn high_rank_update_transpose(
        &mut self,
        alpha: Number,
        v1: &crate::multi_vector_matrix::MultiVectorMatrix,
        v2: &crate::multi_vector_matrix::MultiVectorMatrix,
        beta: Number,
    ) {
        debug_assert_eq!(self.space.n_rows, v1.n_cols());
        debug_assert_eq!(self.space.n_cols, v2.n_cols());
        debug_assert!(beta == 0.0 || self.initialized.get());
        let nr = self.nr();
        let nc = self.nc();
        if beta == 0.0 {
            for j in 0..nc {
                let v2j = v2.get_vector(j as Index).as_ref();
                for i in 0..nr {
                    let v1i = v1.get_vector(i as Index).as_ref();
                    self.values[i + j * nr] = alpha * v1i.dot(v2j);
                }
            }
        } else {
            for j in 0..nc {
                let v2j = v2.get_vector(j as Index).as_ref();
                for i in 0..nr {
                    let v1i = v1.get_vector(i as Index).as_ref();
                    self.values[i + j * nr] = alpha * v1i.dot(v2j) + beta * self.values[i + j * nr];
                }
            }
        }
        self.initialized.set(true);
        self.cache.bump();
    }
}

impl TaggedObject for DenseGenMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

fn dense_x_values(x: &dyn Vector) -> Vec<Number> {
    match x.as_any().downcast_ref::<DenseVector>() {
        Some(d) => d.expanded_values(),
        None => panic!("DenseGenMatrix expects a DenseVector argument"),
    }
}

fn dense_y_mut(y: &mut dyn Vector) -> &mut DenseVector {
    match y.as_any_mut().downcast_mut::<DenseVector>() {
        Some(d) => d,
        None => panic!("DenseGenMatrix expects a DenseVector destination"),
    }
}

impl Matrix for DenseGenMatrix {
    fn n_rows(&self) -> Index {
        self.space.n_rows
    }
    fn n_cols(&self) -> Index {
        self.space.n_cols
    }
    fn cache(&self) -> &MatrixCache {
        &self.cache
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_tagged(&self) -> &dyn TaggedObject {
        self
    }
    fn as_dyn_matrix(&self) -> &dyn Matrix {
        self
    }

    /// `y ← α · M · x + β · y`. Reference DGEMV column-outer order:
    /// `for j: temp = α x[j]; for i: y[i] += temp · A[i + j·m]`.
    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        debug_assert!(self.initialized.get());
        let nr = self.nr();
        let nc = self.nc();
        let xvals = dense_x_values(x);
        let dy = dense_y_mut(y);
        let yvals = dy.values_mut();

        if beta == 0.0 {
            for v in yvals.iter_mut() {
                *v = 0.0;
            }
        } else if beta != 1.0 {
            for v in yvals.iter_mut() {
                *v *= beta;
            }
        }
        if alpha == 0.0 {
            return;
        }
        for (j, &xj) in xvals.iter().enumerate().take(nc) {
            let temp = alpha * xj;
            if temp == 0.0 {
                continue;
            }
            let col = &self.values[j * nr..j * nr + nr];
            for (yi, &cij) in yvals.iter_mut().zip(col.iter()) {
                *yi += temp * cij;
            }
        }
    }

    /// `y ← α · Mᵀ · x + β · y`. Reference DGEMV transposed: outer `j`
    /// computes the dot of column `j` of `M` with `x`, accumulates into
    /// `y[j]`.
    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        debug_assert!(self.initialized.get());
        let nr = self.nr();
        let nc = self.nc();
        let xvals = dense_x_values(x);
        let dy = dense_y_mut(y);
        let yvals = dy.values_mut();

        if beta == 0.0 {
            for v in yvals.iter_mut() {
                *v = 0.0;
            }
        } else if beta != 1.0 {
            for v in yvals.iter_mut() {
                *v *= beta;
            }
        }
        if alpha == 0.0 {
            return;
        }
        for (j, yj) in yvals.iter_mut().enumerate().take(nc) {
            let col = &self.values[j * nr..j * nr + nr];
            let mut temp: Number = 0.0;
            for (&cij, &xi) in col.iter().zip(xvals.iter()) {
                temp += cij * xi;
            }
            *yj += alpha * temp;
        }
    }

    fn has_valid_numbers_impl(&self) -> bool {
        debug_assert!(self.initialized.get());
        let mut sum: Number = 0.0;
        for &v in &self.values {
            sum += v.abs();
        }
        sum.is_finite()
    }

    /// `rows_norms[i] = max(rows_norms[i], maxⱼ |M[i,j]|)`. Caller has
    /// already zeroed if `init`. Iteration order matches upstream (row
    /// outer, col inner) but storage is column-major so we do strided
    /// reads — same arithmetic, no new fp ops vs. abs/max comparisons.
    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        debug_assert!(self.initialized.get());
        let nr = self.nr();
        let nc = self.nc();
        let dy = dense_y_mut(rows_norms);
        let vec_vals = dy.values_mut();
        for (irow, vi) in vec_vals.iter_mut().enumerate().take(nr) {
            for jcol in 0..nc {
                let f = self.values[irow + jcol * nr].abs();
                if f > *vi {
                    *vi = f;
                }
            }
        }
    }

    /// `cols_norms[j] = max(cols_norms[j], maxᵢ |M[i,j]|)`. Walks each
    /// column with an `iamax`-style first-of-equals tie-break — matches
    /// `IpBlasIamax`.
    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, _init: bool) {
        debug_assert!(self.initialized.get());
        let nr = self.nr();
        let nc = self.nc();
        let dy = dense_y_mut(cols_norms);
        let vec_vals = dy.values_mut();
        for (jcol, vj) in vec_vals.iter_mut().enumerate().take(nc) {
            let col = &self.values[jcol * nr..jcol * nr + nr];
            let i = crate::blas1::iamax(col, 1, nr as Index) as usize;
            let f = col[i].abs();
            if f > *vj {
                *vj = f;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_vector::DenseVectorSpace;

    fn make_matrix(nr: Index, nc: Index, col_major: &[Number]) -> DenseGenMatrix {
        let space = DenseGenMatrixSpace::new(nr, nc);
        let mut m = space.make_new_dense_gen();
        m.values_mut().copy_from_slice(col_major);
        m
    }

    fn dvec(n: Index, vals: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(n);
        let mut v = space.make_new_dense();
        v.set_values(vals);
        v
    }

    #[test]
    fn mult_2x3_basic() {
        // M = [[1, 2, 3], [4, 5, 6]]   stored column-major: 1,4, 2,5, 3,6
        let m = make_matrix(2, 3, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let x = dvec(3, &[1.0, 1.0, 1.0]);
        let mut y = dvec(2, &[0.0, 0.0]);
        m.mult_vector(1.0, &x, 0.0, &mut y);
        // M * [1,1,1]ᵀ = [6, 15]
        assert_eq!(y.expanded_values(), vec![6.0, 15.0]);

        // β·y + α·Mx with non-trivial scalars
        let mut y2 = dvec(2, &[10.0, 20.0]);
        m.mult_vector(2.0, &x, 0.5, &mut y2);
        // 0.5*[10,20] + 2*[6,15] = [17, 40]
        assert_eq!(y2.expanded_values(), vec![17.0, 40.0]);
    }

    #[test]
    fn trans_mult_2x3_basic() {
        // Same M; Mᵀ * [1,1]ᵀ = [5, 7, 9]
        let m = make_matrix(2, 3, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let x = dvec(2, &[1.0, 1.0]);
        let mut y = dvec(3, &[0.0, 0.0, 0.0]);
        m.trans_mult_vector(1.0, &x, 0.0, &mut y);
        assert_eq!(y.expanded_values(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn fill_identity_and_mult() {
        let space = DenseGenMatrixSpace::new(3, 3);
        let mut m = space.make_new_dense_gen();
        m.fill_identity(2.5);
        let x = dvec(3, &[1.0, 2.0, 3.0]);
        let mut y = dvec(3, &[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, &x, 0.0, &mut y);
        assert_eq!(y.expanded_values(), vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn copy_from_clones_storage() {
        let a = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let space = DenseGenMatrixSpace::new(2, 2);
        let mut b = space.make_new_dense_gen();
        b.copy_from(&a);
        assert_eq!(b.values(), a.values());
    }

    #[test]
    fn scale_columns() {
        let mut m = make_matrix(2, 3, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let s = dvec(3, &[2.0, 3.0, 4.0]);
        m.scale_columns(&s);
        // Cols become [2,2], [3,3], [4,4]
        assert_eq!(m.values(), &[2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn row_col_amax() {
        let m = make_matrix(2, 3, &[1.0, -4.0, 2.0, 5.0, -3.0, 6.0]);
        let mut row_norms = dvec(2, &[0.0, 0.0]);
        m.compute_row_amax(&mut row_norms, true);
        // Row 0 abs: |1|,|2|,|-3| → 3 ; Row 1: |-4|,|5|,|6| → 6
        assert_eq!(row_norms.expanded_values(), vec![3.0, 6.0]);

        let mut col_norms = dvec(3, &[0.0, 0.0, 0.0]);
        m.compute_col_amax(&mut col_norms, true);
        assert_eq!(col_norms.expanded_values(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn has_valid_numbers_detects_nan() {
        let mut m = make_matrix(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(m.has_valid_numbers());
        m.values_mut()[2] = f64::NAN;
        assert!(!m.has_valid_numbers());
    }

    #[test]
    fn cholesky_factor_recovers_l_l_t() {
        // M = [[4, 2], [2, 5]]; L = [[2, 0], [1, 2]]; verify L Lᵀ == M.
        let sym_space = crate::dense_sym_matrix::DenseSymMatrixSpace::new(2);
        let mut m_sym = sym_space.make_new_dense_sym();
        m_sym.values_mut().copy_from_slice(&[4.0, 2.0, 0.0, 5.0]);
        let space = DenseGenMatrixSpace::new(2, 2);
        let mut l = space.make_new_dense_gen();
        let ok = l.compute_cholesky_factor(&m_sym);
        assert!(ok);
        // Expected L stored column-major (lower-triangle, upper zero):
        // col 0: [2, 1]; col 1: [0, 2]
        assert!((l.values()[0] - 2.0).abs() < 1e-12);
        assert!((l.values()[1] - 1.0).abs() < 1e-12);
        assert!((l.values()[2] - 0.0).abs() < 1e-12);
        assert!((l.values()[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_factor_rejects_indefinite() {
        // M = [[1, 2], [2, 1]] — eigenvalues -1, 3 → not PD.
        let sym_space = crate::dense_sym_matrix::DenseSymMatrixSpace::new(2);
        let mut m_sym = sym_space.make_new_dense_sym();
        m_sym.values_mut().copy_from_slice(&[1.0, 2.0, 0.0, 1.0]);
        let space = DenseGenMatrixSpace::new(2, 2);
        let mut l = space.make_new_dense_gen();
        let ok = l.compute_cholesky_factor(&m_sym);
        assert!(!ok);
    }

    #[test]
    fn cholesky_solve_vector_round_trip() {
        // M = [[4, 2], [2, 5]]; b = [6, 7]; solve x = M⁻¹ b.
        // Expected x: M·x = b → x = (1, 1)ᵀ since [[4,2],[2,5]]·[1,1] = [6, 7].
        let sym_space = crate::dense_sym_matrix::DenseSymMatrixSpace::new(2);
        let mut m_sym = sym_space.make_new_dense_sym();
        m_sym.values_mut().copy_from_slice(&[4.0, 2.0, 0.0, 5.0]);
        let space = DenseGenMatrixSpace::new(2, 2);
        let mut l = space.make_new_dense_gen();
        l.compute_cholesky_factor(&m_sym);
        let mut b = dvec(2, &[6.0, 7.0]);
        l.cholesky_solve_vector(&mut b);
        assert!((b.expanded_values()[0] - 1.0).abs() < 1e-12);
        assert!((b.expanded_values()[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_solve_matrix_two_rhs() {
        let sym_space = crate::dense_sym_matrix::DenseSymMatrixSpace::new(2);
        let mut m_sym = sym_space.make_new_dense_sym();
        m_sym.values_mut().copy_from_slice(&[4.0, 2.0, 0.0, 5.0]);
        let space = DenseGenMatrixSpace::new(2, 2);
        let mut l = space.make_new_dense_gen();
        l.compute_cholesky_factor(&m_sym);
        // RHS columns: [6, 7] and [10, 13] → expected [1, 1] and [2, ?].
        // Verify [10, 13]: [[4,2],[2,5]]·x = [10, 13]; det=16; x = ((5*10-2*13)/16, (-2*10+4*13)/16) = (24/16, 32/16) = (1.5, 2)
        let mut b = make_matrix(2, 2, &[6.0, 7.0, 10.0, 13.0]);
        l.cholesky_solve_matrix(&mut b);
        let v = b.values();
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert!((v[1] - 1.0).abs() < 1e-12);
        assert!((v[2] - 1.5).abs() < 1e-12);
        assert!((v[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_back_solve_forward() {
        // Solve L·x = b. L = [[2,0],[1,2]]; b = [4, 5] → x = [2, 1.5]
        let sym_space = crate::dense_sym_matrix::DenseSymMatrixSpace::new(2);
        let mut m_sym = sym_space.make_new_dense_sym();
        m_sym.values_mut().copy_from_slice(&[4.0, 2.0, 0.0, 5.0]);
        let space = DenseGenMatrixSpace::new(2, 2);
        let mut l = space.make_new_dense_gen();
        l.compute_cholesky_factor(&m_sym);
        let mut b = make_matrix(2, 1, &[4.0, 5.0]);
        l.cholesky_back_solve_matrix(false, 1.0, &mut b);
        let v = b.values();
        assert!((v[0] - 2.0).abs() < 1e-12);
        assert!((v[1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn high_rank_update_transpose_dot_grid() {
        use crate::multi_vector_matrix::MultiVectorMatrixSpace;
        // V1 has 2 columns of dim 3: [1,2,3] and [4,5,6]
        // V2 has 3 columns of dim 3: e1, e2, e3
        // V1ᵀ V2: row i is V1[:,i]; row 0 = [1,2,3]; row 1 = [4,5,6]
        let cs = DenseVectorSpace::new(3);
        let v1_space = MultiVectorMatrixSpace::new(2, Rc::clone(&cs));
        let mut v1 = v1_space.make_new_multi_vector();
        let c0 = {
            let mut v = cs.make_new_dense();
            v.set_values(&[1.0, 2.0, 3.0]);
            std::rc::Rc::new(v)
        };
        let c1 = {
            let mut v = cs.make_new_dense();
            v.set_values(&[4.0, 5.0, 6.0]);
            std::rc::Rc::new(v)
        };
        v1.set_vector(0, c0 as Rc<dyn Vector>);
        v1.set_vector(1, c1 as Rc<dyn Vector>);

        let v2_space = MultiVectorMatrixSpace::new(3, Rc::clone(&cs));
        let mut v2 = v2_space.make_new_multi_vector();
        for k in 0..3 {
            let mut e = cs.make_new_dense();
            let mut buf = [0.0; 3];
            buf[k] = 1.0;
            e.set_values(&buf);
            v2.set_vector(k as Index, std::rc::Rc::new(e) as Rc<dyn Vector>);
        }

        let space = DenseGenMatrixSpace::new(2, 3);
        let mut m = space.make_new_dense_gen();
        m.high_rank_update_transpose(1.0, &v1, &v2, 0.0);
        // Stored column-major: col 0 = [1, 4]; col 1 = [2, 5]; col 2 = [3, 6]
        assert_eq!(m.values(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
