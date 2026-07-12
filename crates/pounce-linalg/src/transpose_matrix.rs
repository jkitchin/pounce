//! Implicit transpose wrapper.
//!
//! Mirrors `LinAlg/IpTransposeMatrix.{hpp,cpp}`. A `TransposeMatrix`
//! owns a reference to an `orig_matrix` and swaps `mult` / `trans_mult`
//! so callers can treat the result as `Mᵀ` without materialising the
//! transpose. Likewise, row/col norms are swapped.

use crate::matrix::{Matrix, MatrixCache};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

#[derive(Debug)]
pub struct TransposeMatrix {
    orig: Rc<dyn Matrix>,
    cache: MatrixCache,
}

impl TransposeMatrix {
    pub fn new(orig: Rc<dyn Matrix>) -> Self {
        Self {
            orig,
            cache: MatrixCache::new(),
        }
    }

    pub fn orig(&self) -> &Rc<dyn Matrix> {
        &self.orig
    }
}

impl TaggedObject for TransposeMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for TransposeMatrix {
    fn n_rows(&self) -> Index {
        self.orig.n_cols()
    }
    fn n_cols(&self) -> Index {
        self.orig.n_rows()
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
        self.orig.trans_mult_vector(alpha, x, beta, y);
    }

    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        self.orig.mult_vector(alpha, x, beta, y);
    }

    fn has_valid_numbers_impl(&self) -> bool {
        self.orig.has_valid_numbers()
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, init: bool) {
        // Note: we forward to the *public* `compute_col_amax` (not its
        // _impl) on the original matrix so that the wrapper's
        // zero-init step still runs upstream. But upstream's
        // `TransposeMatrix::ComputeRowAMaxImpl` calls
        // `orig_matrix_->ComputeColAMax(rows_norms, init)` (the public
        // wrapper) — and our public wrapper has *already* zeroed
        // rows_norms based on `init`. Calling the public form again
        // would re-zero — so forward to the inner _impl directly.
        // This matches upstream's behaviour bit-for-bit (only one
        // zero-init occurs per call to the outer-most public method).
        self.orig.compute_col_amax_impl(rows_norms, init);
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        self.orig.compute_row_amax_impl(cols_norms, init);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseVector;
    use crate::dense_vector::DenseVectorSpace;
    use crate::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};

    fn dvec_box(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    #[test]
    fn transpose_swaps_mult_and_trans_mult() {
        // Original P maps small (2) → large (5), rows {1,3}.
        let exp_space = ExpansionMatrixSpace::new(5, 2, &[1, 3], 0);
        let p: Rc<dyn Matrix> = Rc::new(ExpansionMatrix::new(exp_space));
        let pt = TransposeMatrix::new(Rc::clone(&p));

        // Pᵀ has shape 2×5. Pᵀ * (large) = filter-down.
        assert_eq!(pt.n_rows(), 2);
        assert_eq!(pt.n_cols(), 5);

        let large = dvec_box(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut small = dvec_box(&[0.0, 0.0]);
        pt.mult_vector(1.0, large.as_dyn_vector(), 0.0, small.as_mut());
        let dv = small.as_any().downcast_ref::<DenseVector>().unwrap();
        // Same as P^T * large = [20, 40].
        assert_eq!(dv.expanded_values().to_vec(), vec![20.0, 40.0]);

        // pt.trans_mult should = p.mult: lift small back into large.
        let small2 = dvec_box(&[7.0, -2.0]);
        let mut large2 = dvec_box(&[0.0; 5]);
        pt.trans_mult_vector(1.0, small2.as_dyn_vector(), 0.0, large2.as_mut());
        let dv = large2.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(
            dv.expanded_values().to_vec(),
            vec![0.0, 7.0, 0.0, -2.0, 0.0]
        );
    }
}
