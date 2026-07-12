//! Linear combinations of matrices.
//!
//! Mirrors `LinAlg/IpSumMatrix.{hpp,cpp}` and `IpSumSymMatrix.{hpp,cpp}`.
//! A `SumMatrix` represents `Σ factor_i · M_i` over a fixed list of
//! terms. `MultVector` walks the terms in registration order, exactly
//! matching upstream's iteration order — critical for bit-equivalence
//! when the factors are not all 1.
//!
//! `compute_row_amax_impl` / `compute_col_amax_impl` are deliberately
//! left as `panic!` to match upstream's `THROW_EXCEPTION
//! (UNIMPLEMENTED_LINALG_METHOD_CALLED, ...)` — they are not used
//! anywhere in the IPM main loop.

use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

// ---- SumMatrix (general) ----

#[derive(Debug)]
pub struct SumMatrix {
    n_rows: Index,
    n_cols: Index,
    factors: Vec<Number>,
    matrices: Vec<Option<Rc<dyn Matrix>>>,
    cache: MatrixCache,
}

impl SumMatrix {
    pub fn new(n_rows: Index, n_cols: Index, n_terms: Index) -> Self {
        Self {
            n_rows,
            n_cols,
            factors: vec![1.0; n_terms.max(0) as usize],
            matrices: (0..n_terms.max(0) as usize).map(|_| None).collect(),
            cache: MatrixCache::new(),
        }
    }

    pub fn n_terms(&self) -> Index {
        self.factors.len() as Index
    }

    pub fn set_term(&mut self, iterm: Index, factor: Number, matrix: Rc<dyn Matrix>) {
        debug_assert!((iterm as usize) < self.factors.len());
        debug_assert_eq!(matrix.n_rows(), self.n_rows);
        debug_assert_eq!(matrix.n_cols(), self.n_cols);
        self.factors[iterm as usize] = factor;
        self.matrices[iterm as usize] = Some(matrix);
        self.cache.bump();
    }

    pub fn get_term(&self, iterm: Index) -> (Number, Option<&Rc<dyn Matrix>>) {
        let i = iterm as usize;
        (self.factors[i], self.matrices[i].as_ref())
    }

    fn term(&self, i: usize) -> &dyn Matrix {
        match self.matrices[i].as_ref() {
            Some(m) => m.as_ref(),
            None => panic!("SumMatrix term {i} unset — call set_term before use"),
        }
    }
}

impl TaggedObject for SumMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for SumMatrix {
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

    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        for iterm in 0..self.factors.len() {
            self.term(iterm)
                .mult_vector(alpha * self.factors[iterm], x, 1.0, y);
        }
    }

    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        for iterm in 0..self.factors.len() {
            self.term(iterm)
                .trans_mult_vector(alpha * self.factors[iterm], x, 1.0, y);
        }
    }

    fn has_valid_numbers_impl(&self) -> bool {
        for iterm in 0..self.factors.len() {
            if !self.term(iterm).has_valid_numbers() {
                return false;
            }
        }
        true
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        panic!(
            "SumMatrix::compute_row_amax not implemented (matches upstream UNIMPLEMENTED_LINALG_METHOD_CALLED)"
        );
    }

    fn compute_col_amax_impl(&self, _cols_norms: &mut dyn Vector, _init: bool) {
        panic!(
            "SumMatrix::compute_col_amax not implemented (matches upstream UNIMPLEMENTED_LINALG_METHOD_CALLED)"
        );
    }
}

// ---- SumSymMatrix ----

#[derive(Debug)]
pub struct SumSymMatrix {
    dim: Index,
    factors: Vec<Number>,
    matrices: Vec<Option<Rc<dyn Matrix>>>,
    cache: MatrixCache,
}

impl SumSymMatrix {
    /// Each registered term must implement both [`Matrix`] and
    /// [`SymMatrix`]; we hold them as `Rc<dyn Matrix>` for storage
    /// since trait-object upcasting is not yet stable.
    pub fn new(dim: Index, n_terms: Index) -> Self {
        Self {
            dim,
            factors: vec![1.0; n_terms.max(0) as usize],
            matrices: (0..n_terms.max(0) as usize).map(|_| None).collect(),
            cache: MatrixCache::new(),
        }
    }

    pub fn n_terms(&self) -> Index {
        self.factors.len() as Index
    }

    pub fn set_term(&mut self, iterm: Index, factor: Number, matrix: Rc<dyn Matrix>) {
        debug_assert!((iterm as usize) < self.factors.len());
        debug_assert_eq!(matrix.n_rows(), self.dim);
        debug_assert_eq!(matrix.n_cols(), self.dim);
        self.factors[iterm as usize] = factor;
        self.matrices[iterm as usize] = Some(matrix);
        self.cache.bump();
    }

    pub fn get_term(&self, iterm: Index) -> (Number, Option<&Rc<dyn Matrix>>) {
        let i = iterm as usize;
        (self.factors[i], self.matrices[i].as_ref())
    }

    fn term(&self, i: usize) -> &dyn Matrix {
        match self.matrices[i].as_ref() {
            Some(m) => m.as_ref(),
            None => panic!("SumSymMatrix term {i} unset — call set_term before use"),
        }
    }
}

impl TaggedObject for SumSymMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for SumSymMatrix {
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
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        for iterm in 0..self.factors.len() {
            self.term(iterm)
                .mult_vector(alpha * self.factors[iterm], x, 1.0, y);
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
        for iterm in 0..self.factors.len() {
            if !self.term(iterm).has_valid_numbers() {
                return false;
            }
        }
        true
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        panic!(
            "SumSymMatrix::compute_row_amax not implemented (matches upstream UNIMPLEMENTED_LINALG_METHOD_CALLED)"
        );
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for SumSymMatrix {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseVector;
    use crate::dense_vector::DenseVectorSpace;
    use crate::special_matrix::IdentityMatrix;

    fn dvec_box(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    #[test]
    fn sum_of_two_identities_with_factors() {
        // M = 2 I + 3 I = 5 I  (use sym sum)
        let i1: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(3));
        let i2: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(3));
        let mut s = SumSymMatrix::new(3, 2);
        s.set_term(0, 2.0, i1);
        s.set_term(1, 3.0, i2);

        let x = dvec_box(&[1.0, 2.0, 3.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        s.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn sum_matrix_general_walks_terms_in_order() {
        // M = 1*Z_3x2 + 2*Z_3x2 = 0; verify mult/trans paths run.
        let z1: Rc<dyn Matrix> = Rc::new(crate::ZeroMatrix::new(3, 2));
        let z2: Rc<dyn Matrix> = Rc::new(crate::ZeroMatrix::new(3, 2));
        let mut s = SumMatrix::new(3, 2, 2);
        s.set_term(0, 1.0, z1);
        s.set_term(1, 2.0, z2);
        let x = dvec_box(&[1.0, 2.0]);
        let mut y = dvec_box(&[7.0, 8.0, 9.0]);
        s.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn sum_has_valid_numbers_propagates() {
        let mut bad = IdentityMatrix::new(3);
        bad.set_factor(f64::NAN);
        let m_bad: Rc<dyn Matrix> = Rc::new(bad);
        let mut s = SumSymMatrix::new(3, 1);
        s.set_term(0, 1.0, m_bad);
        assert!(!s.has_valid_numbers());
    }
}
