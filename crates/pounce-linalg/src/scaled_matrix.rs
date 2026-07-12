//! Diagonal-scaling wrappers.
//!
//! Mirrors `LinAlg/IpScaledMatrix.{hpp,cpp}` and
//! `IpSymScaledMatrix.{hpp,cpp}`. `ScaledMatrix` represents
//! `diag(r) · A · diag(c)` where `r` and `c` are optional row/column
//! scaling vectors. `SymScaledMatrix` is the symmetric refinement,
//! `diag(d) · A · diag(d)`.
//!
//! `compute_row_amax_impl` / `compute_col_amax_impl` panic to mirror
//! upstream `THROW_EXCEPTION(UNIMPLEMENTED_LINALG_METHOD_CALLED, ...)`.

use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

// ---- ScaledMatrix ----

/// Reciprocal flag for [`ScaledMatrixSpace::new`].
#[derive(Copy, Clone, Debug)]
pub enum ScalingReciprocal {
    Direct,
    Reciprocal,
}

/// Scaling-vector ownership for a [`ScaledMatrix`]. Optional because
/// upstream allows the row or column scaling to be NULL — meaning that
/// side is not scaled at all.
#[derive(Debug)]
pub struct ScaledMatrixSpace {
    n_rows: Index,
    n_cols: Index,
    row_scaling: Option<Rc<dyn Vector>>,
    col_scaling: Option<Rc<dyn Vector>>,
}

impl ScaledMatrixSpace {
    /// Build a scaling-space. The reciprocal flags follow upstream's
    /// constructor: when set, the scaling vector is *copied and
    /// element-wise-reciprocated* up front, so matrix-vector products
    /// then just multiply (no per-call reciprocal).
    pub fn new(
        n_rows: Index,
        n_cols: Index,
        row_scaling: Option<Rc<dyn Vector>>,
        row_recip: ScalingReciprocal,
        col_scaling: Option<Rc<dyn Vector>>,
        col_recip: ScalingReciprocal,
    ) -> Rc<Self> {
        let row = row_scaling.map(|r| {
            let mut copy = r.make_new_copy();
            if matches!(row_recip, ScalingReciprocal::Reciprocal) {
                copy.element_wise_reciprocal();
            }
            // Box<dyn Vector> → Rc<dyn Vector>; storage handoff.
            let r: Rc<dyn Vector> = Rc::from(copy);
            r
        });
        let col = col_scaling.map(|c| {
            let mut copy = c.make_new_copy();
            if matches!(col_recip, ScalingReciprocal::Reciprocal) {
                copy.element_wise_reciprocal();
            }
            let c: Rc<dyn Vector> = Rc::from(copy);
            c
        });
        Rc::new(Self {
            n_rows,
            n_cols,
            row_scaling: row,
            col_scaling: col,
        })
    }

    pub fn row_scaling(&self) -> Option<&Rc<dyn Vector>> {
        self.row_scaling.as_ref()
    }
    pub fn col_scaling(&self) -> Option<&Rc<dyn Vector>> {
        self.col_scaling.as_ref()
    }
    pub fn n_rows(&self) -> Index {
        self.n_rows
    }
    pub fn n_cols(&self) -> Index {
        self.n_cols
    }
}

#[derive(Debug)]
pub struct ScaledMatrix {
    space: Rc<ScaledMatrixSpace>,
    matrix: Option<Rc<dyn Matrix>>,
    cache: MatrixCache,
}

impl ScaledMatrix {
    pub fn new(space: Rc<ScaledMatrixSpace>) -> Self {
        Self {
            space,
            matrix: None,
            cache: MatrixCache::new(),
        }
    }

    pub fn set_unscaled(&mut self, m: Rc<dyn Matrix>) {
        debug_assert_eq!(m.n_rows(), self.space.n_rows);
        debug_assert_eq!(m.n_cols(), self.space.n_cols);
        self.matrix = Some(m);
        self.cache.bump();
    }

    pub fn unscaled(&self) -> Option<&Rc<dyn Matrix>> {
        self.matrix.as_ref()
    }

    fn inner(&self) -> &dyn Matrix {
        match self.matrix.as_ref() {
            Some(m) => m.as_ref(),
            None => panic!("ScaledMatrix unscaled matrix unset — call set_unscaled before use"),
        }
    }
}

impl TaggedObject for ScaledMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for ScaledMatrix {
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

    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        // Upstream sequence (IpScaledMatrix.cpp:22-58):
        // 1. y *= beta (or y = 0)
        // 2. tmp_x = x.MakeNewCopy()
        // 3. if col scaling: tmp_x .*= col_scaling
        // 4. tmp_y = y.MakeNew(); A * tmp_x → tmp_y (beta=0)
        // 5. if row scaling: tmp_y .*= row_scaling
        // 6. y.Axpy(alpha, tmp_y)
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        let mut tmp_x = x.make_new_copy();
        let mut tmp_y = y.make_new();
        if let Some(c) = self.space.col_scaling() {
            tmp_x.element_wise_multiply(c.as_ref());
        }
        self.inner()
            .mult_vector(1.0, tmp_x.as_dyn_vector(), 0.0, tmp_y.as_mut());
        if let Some(r) = self.space.row_scaling() {
            tmp_y.element_wise_multiply(r.as_ref());
        }
        y.axpy(alpha, tmp_y.as_dyn_vector());
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
        let mut tmp_x = x.make_new_copy();
        let mut tmp_y = y.make_new();
        if let Some(r) = self.space.row_scaling() {
            tmp_x.element_wise_multiply(r.as_ref());
        }
        self.inner()
            .trans_mult_vector(1.0, tmp_x.as_dyn_vector(), 0.0, tmp_y.as_mut());
        if let Some(c) = self.space.col_scaling() {
            tmp_y.element_wise_multiply(c.as_ref());
        }
        y.axpy(alpha, tmp_y.as_dyn_vector());
    }

    fn has_valid_numbers_impl(&self) -> bool {
        self.inner().has_valid_numbers()
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        panic!(
            "ScaledMatrix::compute_row_amax not implemented (matches upstream UNIMPLEMENTED_LINALG_METHOD_CALLED)"
        );
    }

    fn compute_col_amax_impl(&self, _cols_norms: &mut dyn Vector, _init: bool) {
        panic!(
            "ScaledMatrix::compute_col_amax not implemented (matches upstream UNIMPLEMENTED_LINALG_METHOD_CALLED)"
        );
    }
}

// ---- SymScaledMatrix ----

#[derive(Debug)]
pub struct SymScaledMatrixSpace {
    dim: Index,
    row_col_scaling: Option<Rc<dyn Vector>>,
}

impl SymScaledMatrixSpace {
    pub fn new(
        dim: Index,
        row_col_scaling: Option<Rc<dyn Vector>>,
        recip: ScalingReciprocal,
    ) -> Rc<Self> {
        let s = row_col_scaling.map(|s| {
            let mut copy = s.make_new_copy();
            if matches!(recip, ScalingReciprocal::Reciprocal) {
                copy.element_wise_reciprocal();
            }
            let s: Rc<dyn Vector> = Rc::from(copy);
            s
        });
        Rc::new(Self {
            dim,
            row_col_scaling: s,
        })
    }

    pub fn row_col_scaling(&self) -> Option<&Rc<dyn Vector>> {
        self.row_col_scaling.as_ref()
    }
    pub fn dim(&self) -> Index {
        self.dim
    }
}

#[derive(Debug)]
pub struct SymScaledMatrix {
    space: Rc<SymScaledMatrixSpace>,
    matrix: Option<Rc<dyn Matrix>>,
    cache: MatrixCache,
}

impl SymScaledMatrix {
    pub fn new(space: Rc<SymScaledMatrixSpace>) -> Self {
        Self {
            space,
            matrix: None,
            cache: MatrixCache::new(),
        }
    }

    pub fn set_unscaled(&mut self, m: Rc<dyn Matrix>) {
        debug_assert_eq!(m.n_rows(), self.space.dim);
        debug_assert_eq!(m.n_cols(), self.space.dim);
        self.matrix = Some(m);
        self.cache.bump();
    }

    fn inner(&self) -> &dyn Matrix {
        match self.matrix.as_ref() {
            Some(m) => m.as_ref(),
            None => panic!("SymScaledMatrix unscaled matrix unset — call set_unscaled before use"),
        }
    }
}

impl TaggedObject for SymScaledMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for SymScaledMatrix {
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

    fn mult_vector_impl(&self, alpha: Number, x: &dyn Vector, beta: Number, y: &mut dyn Vector) {
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        let mut tmp_x = x.make_new_copy();
        let mut tmp_y = y.make_new();
        if let Some(s) = self.space.row_col_scaling() {
            tmp_x.element_wise_multiply(s.as_ref());
        }
        self.inner()
            .mult_vector(1.0, tmp_x.as_dyn_vector(), 0.0, tmp_y.as_mut());
        if let Some(s) = self.space.row_col_scaling() {
            tmp_y.element_wise_multiply(s.as_ref());
        }
        y.axpy(alpha, tmp_y.as_dyn_vector());
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
        self.inner().has_valid_numbers()
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        panic!(
            "SymScaledMatrix::compute_row_amax not implemented (matches upstream UNIMPLEMENTED_LINALG_METHOD_CALLED)"
        );
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for SymScaledMatrix {}

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

    fn rc_dvec(values: &[Number]) -> Rc<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Rc::new(v)
    }

    #[test]
    fn scaled_matrix_applies_row_and_column_scaling() {
        // R diag(r) * I_factor=1 * diag(c) * x = r .* (c .* x)
        let r = rc_dvec(&[2.0, 3.0]);
        let c = rc_dvec(&[5.0, 7.0]);
        let space = ScaledMatrixSpace::new(
            2,
            2,
            Some(r),
            ScalingReciprocal::Direct,
            Some(c),
            ScalingReciprocal::Direct,
        );
        let mut m = ScaledMatrix::new(space);
        let inner: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(2));
        m.set_unscaled(inner);
        let x = dvec_box(&[1.0, 1.0]);
        let mut y = dvec_box(&[0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        // Expected: r .* (c .* x) = [2,3] .* [5,7] = [10, 21]
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![10.0, 21.0]);
    }

    #[test]
    fn scaled_matrix_with_no_scaling_is_identity_wrapped() {
        let space = ScaledMatrixSpace::new(
            2,
            2,
            None,
            ScalingReciprocal::Direct,
            None,
            ScalingReciprocal::Direct,
        );
        let mut m = ScaledMatrix::new(space);
        let inner: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(2));
        m.set_unscaled(inner);
        let x = dvec_box(&[3.0, 4.0]);
        let mut y = dvec_box(&[10.0, 20.0]);
        m.mult_vector(2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        // y ← 2*I*x + 0.5 y = [6,8] + [5,10] = [11,18]
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![11.0, 18.0]);
    }

    #[test]
    fn sym_scaled_matrix_doubles_scaling() {
        // diag(d) I diag(d) x = d .* d .* x
        let d = rc_dvec(&[2.0, 3.0]);
        let space = SymScaledMatrixSpace::new(2, Some(d), ScalingReciprocal::Direct);
        let mut m = SymScaledMatrix::new(space);
        let inner: Rc<dyn Matrix> = Rc::new(IdentityMatrix::new(2));
        m.set_unscaled(inner);
        let x = dvec_box(&[1.0, 1.0]);
        let mut y = dvec_box(&[0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        // Expected: [4, 9]
        let dv = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.expanded_values().to_vec(), vec![4.0, 9.0]);
    }
}
