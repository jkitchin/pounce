//! Dense symmetric matrix — port of `LinAlg/IpDenseSymMatrix.{hpp,cpp}`.
//!
//! Storage uses BLAS lower-triangular convention in column-major order:
//! the lower triangle of column `j` (rows `i ∈ [j, n)`) lives at
//! `values[i + j·n]`. The strictly upper triangle is unused — all
//! reads/writes go through the symmetric helpers below. Phase 2 ports
//! BLAS-2 only: `mult_vector`, `FillIdentity`, `AddMatrix`, the
//! SR1-helper `SpecialAddForLMSR1` (kept here since it's pure
//! arithmetic; `L` is a `DenseGenMatrix` and we already have one),
//! `compute_row_amax`, `has_valid_numbers`. The `MultiVectorMatrix`-fronted
//! `HighRankUpdateTranspose` is implemented (used by the L-BFGS aug-system
//! solver). The `dsyrk`-fronted dense `HighRankUpdate` variant has no caller
//! in pounce and was never ported — it is intentionally omitted.
//!
//! `mult_vector_impl` uses the reference DSYMV scalar-loop order so
//! results are deterministic across BLAS implementations.

use crate::dense_gen_matrix::DenseGenMatrix;
use crate::dense_vector::DenseVector;
use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug)]
pub struct DenseSymMatrixSpace {
    dim: Index,
}

impl DenseSymMatrixSpace {
    pub fn new(dim: Index) -> Rc<Self> {
        Rc::new(Self { dim })
    }
    pub fn dim(&self) -> Index {
        self.dim
    }
    pub fn make_new_dense_sym(self: &Rc<Self>) -> DenseSymMatrix {
        DenseSymMatrix::new(Rc::clone(self))
    }
}

#[derive(Debug)]
pub struct DenseSymMatrix {
    space: Rc<DenseSymMatrixSpace>,
    cache: MatrixCache,
    /// Column-major storage of size `dim·dim` so we can pass a leading
    /// dimension equal to `dim` to BLAS routines once they're wired.
    /// Only the lower triangle is meaningful.
    values: Vec<Number>,
    initialized: Cell<bool>,
}

impl DenseSymMatrix {
    pub fn new(space: Rc<DenseSymMatrixSpace>) -> Self {
        let n = space.dim.max(0) as usize;
        Self {
            space,
            cache: MatrixCache::new(),
            values: vec![0.0; n * n],
            initialized: Cell::new(false),
        }
    }

    pub fn space(&self) -> &Rc<DenseSymMatrixSpace> {
        &self.space
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.get()
    }

    pub fn values(&self) -> &[Number] {
        debug_assert!(self.initialized.get());
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Number] {
        self.initialized.set(true);
        self.cache.bump();
        &mut self.values
    }

    fn n(&self) -> usize {
        self.space.dim.max(0) as usize
    }

    /// `M ← factor · I` (lower triangle stored).
    pub fn fill_identity(&mut self, factor: Number) {
        let n = self.n();
        for j in 0..n {
            self.values[j + j * n] = factor;
            for i in (j + 1)..n {
                self.values[i + j * n] = 0.0;
            }
        }
        self.initialized.set(true);
        self.cache.bump();
    }

    /// `M ← α·A + β·M` (lower triangle only, but enough for symmetry).
    pub fn add_matrix(&mut self, alpha: Number, a: &DenseSymMatrix, beta: Number) {
        debug_assert!(beta == 0.0 || self.initialized.get());
        debug_assert_eq!(self.space.dim, a.space.dim);
        if alpha == 0.0 {
            if beta == 0.0 {
                let n = self.n();
                for j in 0..n {
                    for i in j..n {
                        self.values[i + j * n] = 0.0;
                    }
                }
                self.initialized.set(true);
                self.cache.bump();
            } else if beta != 1.0 {
                let n = self.n();
                for j in 0..n {
                    for i in j..n {
                        self.values[i + j * n] *= beta;
                    }
                }
                self.cache.bump();
            }
            return;
        }
        let n = self.n();
        let av = a.values();
        if beta == 0.0 {
            for j in 0..n {
                for i in j..n {
                    self.values[i + j * n] = alpha * av[i + j * n];
                }
            }
        } else if beta == 1.0 {
            for j in 0..n {
                for i in j..n {
                    self.values[i + j * n] += alpha * av[i + j * n];
                }
            }
        } else {
            for j in 0..n {
                for i in j..n {
                    self.values[i + j * n] = alpha * av[i + j * n] + beta * self.values[i + j * n];
                }
            }
        }
        self.initialized.set(true);
        self.cache.bump();
    }

    /// `M ← M + D + L + Lᵀ` where `D` is the diagonal in `d` and `L`
    /// is *strictly* lower-triangular (its upper part is ignored).
    /// Used by L-BFGS SR1.
    pub fn special_add_for_lmsr1(&mut self, d: &DenseVector, l: &DenseGenMatrix) {
        let n = self.n();
        debug_assert!(self.initialized.get());
        debug_assert_eq!(d.dim() as usize, n);
        debug_assert_eq!(l.n_rows() as usize, n);
        debug_assert_eq!(l.n_cols() as usize, n);
        let dvals = d.expanded_values();
        for (i, &di) in dvals.iter().enumerate().take(n) {
            self.values[i + i * n] += di;
        }
        let lv = l.values();
        for j in 0..n {
            for i in (j + 1)..n {
                self.values[i + j * n] += lv[i + j * n];
            }
        }
        self.cache.bump();
    }

    /// `M[i, j] ← α · V1[:, i]ᵀ · V2[:, j] + β · M[i, j]` for `j ≤ i`
    /// (lower triangle only). Port of `DenseSymMatrix::HighRankUpdateTranspose`
    /// (`IpDenseSymMatrix.cpp:124-158`). Used by the L-BFGS aug-system
    /// solver to assemble `Vtilde1ᵀ · V_x` and friends.
    pub fn high_rank_update_transpose(
        &mut self,
        alpha: Number,
        v1: &crate::multi_vector_matrix::MultiVectorMatrix,
        v2: &crate::multi_vector_matrix::MultiVectorMatrix,
        beta: Number,
    ) {
        debug_assert_eq!(self.space.dim, v1.n_cols());
        debug_assert_eq!(self.space.dim, v2.n_cols());
        debug_assert!(beta == 0.0 || self.initialized.get());
        let n = self.n();
        if beta == 0.0 {
            for j in 0..n {
                let v2j = v2.get_vector(j as Index).as_ref();
                for i in j..n {
                    let v1i = v1.get_vector(i as Index).as_ref();
                    self.values[i + j * n] = alpha * v1i.dot(v2j);
                }
            }
        } else {
            for j in 0..n {
                let v2j = v2.get_vector(j as Index).as_ref();
                for i in j..n {
                    let v1i = v1.get_vector(i as Index).as_ref();
                    self.values[i + j * n] = alpha * v1i.dot(v2j) + beta * self.values[i + j * n];
                }
            }
        }
        self.initialized.set(true);
        self.cache.bump();
    }
}

impl TaggedObject for DenseSymMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

fn dense_x_values(x: &dyn Vector) -> Vec<Number> {
    match x.as_any().downcast_ref::<DenseVector>() {
        Some(d) => d.expanded_values(),
        None => panic!("DenseSymMatrix expects a DenseVector argument"),
    }
}

fn dense_y_mut(y: &mut dyn Vector) -> &mut DenseVector {
    match y.as_any_mut().downcast_mut::<DenseVector>() {
        Some(d) => d,
        None => panic!("DenseSymMatrix expects a DenseVector destination"),
    }
}

impl Matrix for DenseSymMatrix {
    fn n_rows(&self) -> Index {
        self.space.dim
    }
    fn n_cols(&self) -> Index {
        self.space.dim
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

    /// `y ← α·A·x + β·y` for symmetric `A` (lower triangle stored).
    /// Reference DSYMV ordering: outer `j` updates both `y[j]` (using
    /// the column from row `j` downward) and column `j`'s contribution
    /// to `y[i]` for `i > j` simultaneously.
    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        debug_assert!(self.initialized.get());
        let n = self.n();
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
        for j in 0..n {
            let temp1 = alpha * xvals[j];
            let mut temp2: Number = 0.0;
            yvals[j] += temp1 * self.values[j + j * n];
            // strict lower triangle of column j: A[i,j] for i > j
            let col_lower = &self.values[(j + 1 + j * n)..(j * n + n)];
            let x_lower = &xvals[(j + 1)..n];
            let (yj_block, y_lower) = yvals.split_at_mut(j + 1);
            let _ = yj_block; // silence unused
            for ((y_i, &a_ij), &x_i) in y_lower.iter_mut().zip(col_lower.iter()).zip(x_lower.iter())
            {
                *y_i += temp1 * a_ij;
                temp2 += a_ij * x_i;
            }
            yvals[j] += alpha * temp2;
        }
    }

    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        sym_default_trans_mult_vector_impl(self, alpha, x, beta, y);
    }

    fn has_valid_numbers_impl(&self) -> bool {
        debug_assert!(self.initialized.get());
        let n = self.n();
        let mut sum: Number = 0.0;
        for j in 0..n {
            sum += self.values[j + j * n];
            for i in (j + 1)..n {
                sum += self.values[i + j * n];
            }
        }
        sum.is_finite()
    }

    /// Per upstream: a single row-amax walk over the lower triangle
    /// updates *both* `vec_vals[irow]` and `vec_vals[jcol]` (since by
    /// symmetry `|A[j,i]| = |A[i,j]|`). Caller has already zeroed if
    /// `init`.
    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        debug_assert!(self.initialized.get());
        let n = self.n();
        let dy = dense_y_mut(rows_norms);
        let vec_vals = dy.values_mut();
        for irow in 0..n {
            for jcol in 0..=irow {
                let f = self.values[irow + jcol * n].abs();
                let (a, b) = if irow == jcol {
                    (irow, irow)
                } else {
                    (irow, jcol)
                };
                if f > vec_vals[a] {
                    vec_vals[a] = f;
                }
                if a != b && f > vec_vals[b] {
                    vec_vals[b] = f;
                }
            }
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for DenseSymMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_vector::DenseVectorSpace;

    fn dvec(n: Index, vals: &[Number]) -> DenseVector {
        let space = DenseVectorSpace::new(n);
        let mut v = space.make_new_dense();
        v.set_values(vals);
        v
    }

    /// Helper: write only the lower triangle of an n×n symmetric
    /// matrix from a row-major specification.
    fn make_sym(dim: Index, lower: &[Number]) -> DenseSymMatrix {
        let n = dim as usize;
        debug_assert_eq!(lower.len(), n * n);
        let space = DenseSymMatrixSpace::new(dim);
        let mut m = space.make_new_dense_sym();
        let v = m.values_mut();
        for j in 0..n {
            for i in j..n {
                v[i + j * n] = lower[i * n + j]; // input is row-major
            }
        }
        m
    }

    #[test]
    fn fill_identity_then_mult() {
        let space = DenseSymMatrixSpace::new(3);
        let mut m = space.make_new_dense_sym();
        m.fill_identity(2.0);
        let x = dvec(3, &[1.0, -2.0, 3.0]);
        let mut y = dvec(3, &[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, &x, 0.0, &mut y);
        assert_eq!(y.expanded_values(), vec![2.0, -4.0, 6.0]);
    }

    #[test]
    fn mult_3x3_symmetric_matches_full() {
        // A = [[1,2,3],[2,4,5],[3,5,6]] symmetric
        let m = make_sym(
            3,
            &[
                1.0, 2.0, 3.0, // row 0
                2.0, 4.0, 5.0, // row 1
                3.0, 5.0, 6.0, // row 2
            ],
        );
        let x = dvec(3, &[1.0, 1.0, 1.0]);
        let mut y = dvec(3, &[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, &x, 0.0, &mut y);
        // A * [1,1,1] = [6, 11, 14]
        assert_eq!(y.expanded_values(), vec![6.0, 11.0, 14.0]);

        // Transpose path (just delegates to mult)
        let mut y2 = dvec(3, &[0.0, 0.0, 0.0]);
        m.trans_mult_vector(1.0, &x, 0.0, &mut y2);
        assert_eq!(y2.expanded_values(), vec![6.0, 11.0, 14.0]);
    }

    #[test]
    fn add_matrix_combines() {
        // A = 2I, B = I → 3·B + 2·A = 3I + 4I = 7I
        let space = DenseSymMatrixSpace::new(2);
        let mut a = space.make_new_dense_sym();
        a.fill_identity(2.0);
        let mut b = space.make_new_dense_sym();
        b.fill_identity(1.0);
        b.add_matrix(2.0, &a, 3.0);
        let x = dvec(2, &[1.0, 1.0]);
        let mut y = dvec(2, &[0.0, 0.0]);
        b.mult_vector(1.0, &x, 0.0, &mut y);
        assert_eq!(y.expanded_values(), vec![7.0, 7.0]);
    }

    #[test]
    fn row_amax_uses_symmetry() {
        // A = [[1,-7],[ -7,2]] → row maxes both = 7.
        let m = make_sym(
            2,
            &[
                1.0, -7.0, // row 0
                -7.0, 2.0, // row 1
            ],
        );
        let mut norms = dvec(2, &[0.0, 0.0]);
        m.compute_row_amax(&mut norms, true);
        assert_eq!(norms.expanded_values(), vec![7.0, 7.0]);
        // col amax = row amax for symmetric matrix
        let mut cnorms = dvec(2, &[0.0, 0.0]);
        m.compute_col_amax(&mut cnorms, true);
        assert_eq!(cnorms.expanded_values(), vec![7.0, 7.0]);
    }

    #[test]
    fn special_add_for_lmsr1() {
        // M starts as zero; add D = diag(1,1) and strictly lower L
        // with L[1,0] = 5. Result lower triangle: [[1, _], [5, 1]].
        // mult by [1,1] should give [1*1 + 5*1, 5*1 + 1*1] = [6, 6].
        let space = DenseSymMatrixSpace::new(2);
        let mut m = space.make_new_dense_sym();
        m.fill_identity(0.0); // zero, initialized

        let d = dvec(2, &[1.0, 1.0]);
        let lspace = crate::dense_gen_matrix::DenseGenMatrixSpace::new(2, 2);
        let mut l = lspace.make_new_dense_gen();
        // column-major: L[1,0] sits at index 1
        l.values_mut().copy_from_slice(&[0.0, 5.0, 0.0, 0.0]);

        m.special_add_for_lmsr1(&d, &l);
        let x = dvec(2, &[1.0, 1.0]);
        let mut y = dvec(2, &[0.0, 0.0]);
        m.mult_vector(1.0, &x, 0.0, &mut y);
        assert_eq!(y.expanded_values(), vec![6.0, 6.0]);
    }

    #[test]
    fn has_valid_numbers_detects_nan() {
        let space = DenseSymMatrixSpace::new(2);
        let mut m = space.make_new_dense_sym();
        m.fill_identity(1.0);
        assert!(m.has_valid_numbers());
        m.values_mut()[0] = f64::NAN;
        assert!(!m.has_valid_numbers());
    }
}
