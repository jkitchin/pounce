//! Diagonal matrices.
//!
//! Mirrors `LinAlg/IpDiagMatrix.{hpp,cpp}`. Storage is one
//! [`Vector`] holding the diagonal entries. Symmetric by construction.

use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

/// Diagonal matrix `D = diag(d)` where `d` is stored as a Vector.
///
/// Upstream stores `SmartPtr<const Vector>` and rebinds via `SetDiag`;
/// we hold `Option<Rc<dyn Vector>>` since Rust's borrow rules prefer
/// shared ownership for the diagonal element source.
#[derive(Debug)]
pub struct DiagMatrix {
    dim: Index,
    diag: Option<Rc<dyn Vector>>,
    cache: MatrixCache,
}

impl DiagMatrix {
    pub fn new(dim: Index) -> Self {
        Self {
            dim,
            diag: None,
            cache: MatrixCache::new(),
        }
    }

    pub fn set_diag(&mut self, diag: Rc<dyn Vector>) {
        debug_assert_eq!(diag.dim(), self.dim);
        self.diag = Some(diag);
        self.cache.bump();
    }

    pub fn get_diag(&self) -> Option<&Rc<dyn Vector>> {
        self.diag.as_ref()
    }

    fn diag_ref(&self) -> &dyn Vector {
        match self.diag.as_ref() {
            Some(d) => d.as_ref(),
            None => panic!("DiagMatrix::diag is unset — call set_diag before use"),
        }
    }
}

impl TaggedObject for DiagMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for DiagMatrix {
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
        let diag = self.diag_ref();
        // Match upstream order: scal(beta) → tmp = x .* diag → y += alpha tmp.
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        let mut tmp = y.make_new();
        tmp.copy(x);
        tmp.element_wise_multiply(diag);
        y.axpy(alpha, tmp.as_dyn_vector());
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
        self.diag_ref().has_valid_numbers()
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, init: bool) {
        let diag = self.diag_ref();
        if init {
            rows_norms.copy(diag);
            rows_norms.element_wise_abs();
        } else {
            let mut v = diag.make_new_copy();
            v.element_wise_abs();
            rows_norms.element_wise_max(v.as_dyn_vector());
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for DiagMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseVector;
    use crate::dense_vector::DenseVectorSpace;

    fn dvec(values: &[Number]) -> Rc<DenseVector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Rc::new(v)
    }

    fn dvec_box(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    #[test]
    fn diag_mult_vector_with_beta_scaling() {
        let d: Rc<dyn Vector> = dvec(&[2.0, -3.0, 4.0]);
        let mut m = DiagMatrix::new(3);
        m.set_diag(d);
        let x = dvec_box(&[1.0, 1.0, 1.0]);
        let mut y = dvec_box(&[10.0, 20.0, 30.0]);
        // y ← 2 * D * x + 0.5 y = 2*[2,-3,4] + [5,10,15] = [9,4,23]
        m.mult_vector(2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![9.0, 4.0, 23.0]);
    }

    #[test]
    fn diag_row_amax_takes_abs() {
        let d: Rc<dyn Vector> = dvec(&[2.0, -3.0, 4.0]);
        let mut m = DiagMatrix::new(3);
        m.set_diag(d);
        let mut norms = dvec_box(&[0.0, 0.0, 0.0]);
        m.compute_row_amax(norms.as_mut(), true);
        let dv = norms.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn diag_has_valid_numbers_propagates_from_diag_vector() {
        let d_ok: Rc<dyn Vector> = dvec(&[1.0, 2.0, 3.0]);
        let mut m = DiagMatrix::new(3);
        m.set_diag(d_ok);
        assert!(m.has_valid_numbers());

        let d_bad: Rc<dyn Vector> = dvec(&[1.0, f64::NAN, 3.0]);
        let mut m2 = DiagMatrix::new(3);
        m2.set_diag(d_bad);
        assert!(!m2.has_valid_numbers());
    }
}
