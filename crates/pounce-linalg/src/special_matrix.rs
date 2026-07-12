//! ZeroMatrix, ZeroSymMatrix, IdentityMatrix.
//!
//! Mirrors `LinAlg/IpZeroMatrix.{hpp,cpp}`, `IpZeroSymMatrix.{hpp,cpp}`,
//! `IpIdentityMatrix.{hpp,cpp}`. These three are storage-free utility
//! matrices used by Ipopt for KKT-system blocks that are absent.

use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;

// ---- ZeroMatrix ----

/// `O(m, n) · x = 0`. Storage-free.
#[derive(Debug)]
pub struct ZeroMatrix {
    n_rows: Index,
    n_cols: Index,
    cache: MatrixCache,
}

impl ZeroMatrix {
    pub fn new(n_rows: Index, n_cols: Index) -> Self {
        Self {
            n_rows,
            n_cols,
            cache: MatrixCache::new(),
        }
    }
}

impl TaggedObject for ZeroMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for ZeroMatrix {
    fn n_rows(&self) -> Index {
        self.n_rows
    }
    fn n_cols(&self) -> Index {
        self.n_cols
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

    fn mult_vector_impl(&self, _alpha: Number, _x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
    }

    fn trans_mult_vector_impl(
        &self,
        _alpha: Number,
        _x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        // All zeros — caller has already set rows_norms to 0 if `init`,
        // and since |0| ≤ existing entries, do nothing.
    }

    fn compute_col_amax_impl(&self, _cols_norms: &mut dyn Vector, _init: bool) {
        // Same as above.
    }
}

// ---- ZeroSymMatrix ----

/// `O(n) · x = 0` constrained to be square symmetric. Storage-free.
#[derive(Debug)]
pub struct ZeroSymMatrix {
    dim: Index,
    cache: MatrixCache,
}

impl ZeroSymMatrix {
    pub fn new(dim: Index) -> Self {
        Self {
            dim,
            cache: MatrixCache::new(),
        }
    }
}

impl TaggedObject for ZeroSymMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for ZeroSymMatrix {
    fn n_rows(&self) -> Index {
        self.dim
    }
    fn n_cols(&self) -> Index {
        self.dim
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

    fn mult_vector_impl(&self, _alpha: Number, _x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
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

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {}

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for ZeroSymMatrix {}

// ---- IdentityMatrix ----

/// `I · α · x` (scalar multiple of identity). Mirrors `IpIdentityMatrix`.
#[derive(Debug)]
pub struct IdentityMatrix {
    dim: Index,
    factor: Number,
    cache: MatrixCache,
}

impl IdentityMatrix {
    pub fn new(dim: Index) -> Self {
        Self {
            dim,
            factor: 1.0,
            cache: MatrixCache::new(),
        }
    }

    pub fn factor(&self) -> Number {
        self.factor
    }

    pub fn set_factor(&mut self, factor: Number) {
        self.factor = factor;
        self.cache.bump();
    }
}

impl TaggedObject for IdentityMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for IdentityMatrix {
    fn n_rows(&self) -> Index {
        self.dim
    }
    fn n_cols(&self) -> Index {
        self.dim
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

    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        // y ← (alpha*factor) x + beta y, via add_one_vector.
        y.add_one_vector(alpha * self.factor, x, beta);
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
        self.factor.is_finite()
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, init: bool) {
        // Caller has already zeroed rows_norms if `init`. Upstream
        // sets it to 1 outright if init, else takes element-wise max
        // with a unit vector. Since |factor·1| might differ from 1
        // upstream chose to record the row max-abs as 1 regardless of
        // factor (see IpIdentityMatrix.cpp:48-63 — it builds a `Set(1)`
        // vector). We replicate that exactly.
        if init {
            rows_norms.set(1.0);
        } else {
            let mut v = rows_norms.make_new();
            v.set(1.0);
            rows_norms.element_wise_max(v.as_dyn_vector());
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }

    fn add_m_sinv_z_impl(&self, alpha: Number, s: &dyn Vector, z: &dyn Vector, x: &mut dyn Vector) {
        // Specialised override per IpIdentityMatrix.cpp:82-95.
        // X.AddVectorQuotient(alpha, Z, S, 1.) — note: factor_ is
        // intentionally omitted upstream (the override exists
        // pre-factor; we mirror the bug-for-bug).
        x.add_vector_quotient(alpha, z, s, 1.0);
    }
}

impl SymMatrix for IdentityMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_vector::DenseVectorSpace;
    use std::rc::Rc;

    fn dvec(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let v = Rc::clone(&space).make_new_dense();
        let mut b: Box<dyn Vector> = Box::new(v);
        // Initialise via set_values through downcast.
        if let Some(dv) = b.as_any_mut().downcast_mut::<crate::DenseVector>() {
            dv.set_values(values);
        }
        b
    }

    #[test]
    fn zero_matrix_zeros_or_scales_y() {
        let m = ZeroMatrix::new(3, 2);
        let x = dvec(&[1.0, 2.0]);
        let mut y = dvec(&[10.0, 20.0, 30.0]);
        m.mult_vector(7.0, x.as_dyn_vector(), 0.5, y.as_mut());
        let dv = y.as_any().downcast_ref::<crate::DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![5.0, 10.0, 15.0]);

        // beta == 0.0 path zeros y even if it had garbage.
        let mut y2 = dvec(&[f64::NAN, f64::INFINITY, -1.0]);
        m.mult_vector(7.0, x.as_dyn_vector(), 0.0, y2.as_mut());
        let dv = y2.as_any().downcast_ref::<crate::DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn identity_matrix_with_factor() {
        let mut m = IdentityMatrix::new(3);
        m.set_factor(2.5);
        let x = dvec(&[1.0, 2.0, 3.0]);
        let mut y = dvec(&[10.0, 20.0, 30.0]);
        // y ← 2 * I_2.5 * x + 0.5 y = 5*[1,2,3] + 0.5*[10,20,30] = [10,20,30]
        m.mult_vector(2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        let dv = y.as_any().downcast_ref::<crate::DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn identity_row_amax_is_one_regardless_of_factor() {
        let mut m = IdentityMatrix::new(3);
        m.set_factor(7.5);
        let mut norms = dvec(&[0.0, 0.0, 0.0]);
        m.compute_row_amax(norms.as_mut(), true);
        let dv = norms.as_any().downcast_ref::<crate::DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn identity_has_valid_numbers_detects_nan() {
        let mut m = IdentityMatrix::new(3);
        assert!(m.has_valid_numbers());
        m.set_factor(f64::NAN);
        assert!(!m.has_valid_numbers());
    }

    #[test]
    fn zero_sym_matrix_multiplies_to_zero() {
        let m = ZeroSymMatrix::new(3);
        let x = dvec(&[1.0, 2.0, 3.0]);
        let mut y = dvec(&[10.0, 20.0, 30.0]);
        m.mult_vector(7.0, x.as_dyn_vector(), 0.0, y.as_mut());
        let dv = y.as_any().downcast_ref::<crate::DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![0.0, 0.0, 0.0]);
    }
}
