//! Symmetric matrix represented as a low-rank update — port of
//! `LinAlg/IpLowRankUpdateSymMatrix.{hpp,cpp}`.
//!
//! Stores `M = D + P · (V Vᵀ - U Uᵀ) · Pᵀ` (full diag) or
//! `M = P · (D + V Vᵀ - U Uᵀ) · Pᵀ` (reduced diag), where:
//! * `D` is a diagonal Vector,
//! * `V`, `U` are [`MultiVectorMatrix`] (the curvature pairs from the
//!   L-BFGS rolling window),
//! * `P` is an optional expansion `Matrix` lifting the small-space
//!   low-rank update into the full primal space (`None` means identity).
//!
//! `MultVectorImpl` follows the four-branch flow of upstream:
//! `(P==None) × (V/U set or not)` and `(reduced_diag) × (P set)`.
//! Each branch uses the same allocations and ordering as
//! `IpLowRankUpdateSymMatrix.cpp:26-131` so the BLAS-1 reduction order
//! is preserved.

use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::multi_vector_matrix::MultiVectorMatrix;
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

#[derive(Debug)]
pub struct LowRankUpdateSymMatrixSpace {
    dim: Index,
    /// Optional expansion `P` lifting the small-space update into the
    /// full primal space. `None` ⇔ identity.
    p_lowrank: Option<Rc<dyn Matrix>>,
    /// Whether the diagonal `D` lives in the *small* (low-rank) space
    /// or the full space.
    reduced_diag: bool,
}

impl LowRankUpdateSymMatrixSpace {
    pub fn new(dim: Index, p_lowrank: Option<Rc<dyn Matrix>>, reduced_diag: bool) -> Rc<Self> {
        Rc::new(Self {
            dim,
            p_lowrank,
            reduced_diag,
        })
    }

    pub fn dim(&self) -> Index {
        self.dim
    }
    pub fn p_lowrank(&self) -> Option<&Rc<dyn Matrix>> {
        self.p_lowrank.as_ref()
    }
    pub fn reduced_diag(&self) -> bool {
        self.reduced_diag
    }

    pub fn make_new_low_rank(self: &Rc<Self>) -> LowRankUpdateSymMatrix {
        LowRankUpdateSymMatrix::new(Rc::clone(self))
    }
}

#[derive(Debug)]
pub struct LowRankUpdateSymMatrix {
    space: Rc<LowRankUpdateSymMatrixSpace>,
    cache: MatrixCache,
    d: Option<Rc<dyn Vector>>,
    v: Option<Rc<MultiVectorMatrix>>,
    u: Option<Rc<MultiVectorMatrix>>,
}

impl LowRankUpdateSymMatrix {
    pub fn new(space: Rc<LowRankUpdateSymMatrixSpace>) -> Self {
        Self {
            space,
            cache: MatrixCache::new(),
            d: None,
            v: None,
            u: None,
        }
    }

    pub fn space(&self) -> &Rc<LowRankUpdateSymMatrixSpace> {
        &self.space
    }

    pub fn set_diag(&mut self, d: Rc<dyn Vector>) {
        self.d = Some(d);
        self.cache.bump();
    }
    pub fn get_diag(&self) -> Option<&Rc<dyn Vector>> {
        self.d.as_ref()
    }

    pub fn set_v(&mut self, v: Rc<MultiVectorMatrix>) {
        self.v = Some(v);
        self.cache.bump();
    }
    pub fn get_v(&self) -> Option<&Rc<MultiVectorMatrix>> {
        self.v.as_ref()
    }

    pub fn set_u(&mut self, u: Rc<MultiVectorMatrix>) {
        self.u = Some(u);
        self.cache.bump();
    }
    pub fn get_u(&self) -> Option<&Rc<MultiVectorMatrix>> {
        self.u.as_ref()
    }

    pub fn p_lowrank(&self) -> Option<&Rc<dyn Matrix>> {
        self.space.p_lowrank()
    }
    pub fn reduced_diag(&self) -> bool {
        self.space.reduced_diag()
    }

    fn diag_ref(&self) -> &dyn Vector {
        self.d
            .as_ref()
            .expect("LowRankUpdateSymMatrix: diagonal D not set")
            .as_ref()
    }
}

impl TaggedObject for LowRankUpdateSymMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for LowRankUpdateSymMatrix {
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
        debug_assert_eq!(self.space.dim, x.dim());
        debug_assert_eq!(self.space.dim, y.dim());
        let d = self.diag_ref();

        match self.space.p_lowrank.as_ref() {
            None => {
                // Diagonal part: y ← α D x + β y (matches upstream
                // ordering: explicit beta-branch to avoid double-scal).
                if beta != 0.0 {
                    let mut tmp = x.make_new_copy();
                    tmp.element_wise_multiply(d);
                    y.add_one_vector(alpha, tmp.as_dyn_vector(), beta);
                } else {
                    y.add_one_vector(alpha, x, 0.0);
                    y.element_wise_multiply(d);
                }
                if let Some(v) = self.v.as_ref() {
                    v.lr_mult_vector(alpha, x, 1.0, y);
                }
                if let Some(u) = self.u.as_ref() {
                    u.lr_mult_vector(-alpha, x, 1.0, y);
                }
            }
            Some(p_lr) => {
                if self.space.reduced_diag {
                    // y ← α P (D + V Vᵀ - U Uᵀ) Pᵀ x + β y
                    let mut small_x = self.make_small_vec_for_p_lr(p_lr.as_ref());
                    p_lr.trans_mult_vector(1.0, x, 0.0, small_x.as_mut());
                    let mut small_y = self.make_small_vec_for_p_lr(p_lr.as_ref());
                    small_y.copy(small_x.as_dyn_vector());
                    small_y.element_wise_multiply(d);
                    if let Some(v) = self.v.as_ref() {
                        v.lr_mult_vector(1.0, small_x.as_dyn_vector(), 1.0, small_y.as_mut());
                    }
                    if let Some(u) = self.u.as_ref() {
                        u.lr_mult_vector(-1.0, small_x.as_dyn_vector(), 1.0, small_y.as_mut());
                    }
                    p_lr.mult_vector(alpha, small_y.as_dyn_vector(), beta, y);
                } else {
                    // Diagonal in full space, low-rank in small space.
                    let mut tmp = x.make_new_copy();
                    tmp.element_wise_multiply(d);
                    y.add_one_vector(alpha, tmp.as_dyn_vector(), beta);

                    let mut small_x = self.make_small_vec_for_p_lr(p_lr.as_ref());
                    p_lr.trans_mult_vector(1.0, x, 0.0, small_x.as_mut());
                    let mut small_y = self.make_small_vec_for_p_lr(p_lr.as_ref());
                    if let Some(v) = self.v.as_ref() {
                        v.lr_mult_vector(1.0, small_x.as_dyn_vector(), 0.0, small_y.as_mut());
                    } else {
                        small_y.set(0.0);
                    }
                    if let Some(u) = self.u.as_ref() {
                        u.lr_mult_vector(-1.0, small_x.as_dyn_vector(), 1.0, small_y.as_mut());
                    }
                    p_lr.mult_vector(alpha, small_y.as_dyn_vector(), 1.0, y);
                }
            }
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
        if !self.diag_ref().has_valid_numbers() {
            return false;
        }
        if let Some(v) = self.v.as_ref() {
            if !v.has_valid_numbers() {
                return false;
            }
        }
        if let Some(u) = self.u.as_ref() {
            if !u.has_valid_numbers() {
                return false;
            }
        }
        true
    }

    fn compute_row_amax_impl(&self, _rows_norms: &mut dyn Vector, _init: bool) {
        unimplemented!("LowRankUpdateSymMatrix::compute_row_amax — upstream throws UNIMPLEMENTED");
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for LowRankUpdateSymMatrix {}

impl LowRankUpdateSymMatrix {
    /// Allocate a fresh dense vector compatible with `P_LR`'s small
    /// (column) space — i.e. dimension `P.n_cols()`. We reach for the
    /// trans-mult target vector's space by first creating a probe of
    /// the right dim via D's `make_new` semantics. Since we only have
    /// `&dyn Matrix` and not the column space, we rely on the
    /// invariant that callers set `D` (or any Vector with the small
    /// dim) first; here we simply allocate a `DenseVector` of
    /// `p_lr.n_cols()` length using D's `make_new` only when D's dim
    /// matches the small space (reduced_diag case). For the
    /// non-reduced-diag case we fall back to a fresh DenseVector.
    fn make_small_vec_for_p_lr(&self, p_lr: &dyn Matrix) -> Box<dyn Vector> {
        // Use D's vector space if D's dimension matches the small dim
        // (this is the reduced_diag path). Otherwise allocate a new
        // DenseVector of the right size.
        let small_dim = p_lr.n_cols();
        if let Some(d) = self.d.as_ref() {
            if d.dim() == small_dim {
                return d.make_new();
            }
        }
        // Fallback: brand-new DenseVector of dimension small_dim.
        let space = crate::dense_vector::DenseVectorSpace::new(small_dim);
        Box::new(space.make_new_dense())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseVector;
    use crate::dense_vector::DenseVectorSpace;
    use crate::multi_vector_matrix::MultiVectorMatrixSpace;

    fn dvec(values: &[Number]) -> Rc<DenseVector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Rc::new(v)
    }

    fn dvec_box(values: &[Number]) -> Box<DenseVector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    fn build_mv(cols: &[&[Number]]) -> Rc<MultiVectorMatrix> {
        let n_rows = cols[0].len() as Index;
        let n_cols = cols.len() as Index;
        let cs = DenseVectorSpace::new(n_rows);
        let space = MultiVectorMatrixSpace::new(n_cols, cs);
        let mut mv = space.make_new_multi_vector();
        for (i, c) in cols.iter().enumerate() {
            mv.set_vector(i as Index, dvec(c) as Rc<dyn Vector>);
        }
        Rc::new(mv)
    }

    #[test]
    fn diag_only_no_p_lr() {
        // M = diag(2, 3, 4); x = [1, 1, 1] → M x = [2, 3, 4]
        let space = LowRankUpdateSymMatrixSpace::new(3, None, false);
        let mut m = space.make_new_low_rank();
        m.set_diag(dvec(&[2.0, 3.0, 4.0]) as Rc<dyn Vector>);
        let x = dvec_box(&[1.0, 1.0, 1.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn diag_plus_v_v_t_no_p_lr() {
        // V = [[1, 0]; [0, 1]; [0, 0]] → V Vᵀ = diag(1, 1, 0)
        // D = diag(1, 1, 1) → M = diag(2, 2, 1)
        // x = [10, 20, 30] → M x = [20, 40, 30]
        let space = LowRankUpdateSymMatrixSpace::new(3, None, false);
        let mut m = space.make_new_low_rank();
        m.set_diag(dvec(&[1.0, 1.0, 1.0]) as Rc<dyn Vector>);
        m.set_v(build_mv(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]));
        let x = dvec_box(&[10.0, 20.0, 30.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![20.0, 40.0, 30.0]);
    }

    #[test]
    fn diag_plus_v_minus_u_no_p_lr() {
        // V Vᵀ - U Uᵀ; choose V = e1 col, U = e2 col → diag(1, -1, 0)
        // D = 0 → M = diag(1, -1, 0)
        let space = LowRankUpdateSymMatrixSpace::new(3, None, false);
        let mut m = space.make_new_low_rank();
        m.set_diag(dvec(&[0.0, 0.0, 0.0]) as Rc<dyn Vector>);
        m.set_v(build_mv(&[&[1.0, 0.0, 0.0]]));
        m.set_u(build_mv(&[&[0.0, 1.0, 0.0]]));
        let x = dvec_box(&[7.0, 11.0, 13.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        assert_eq!(y.expanded_values(), vec![7.0, -11.0, 0.0]);
    }

    #[test]
    fn alpha_beta_combine_with_existing_y() {
        let space = LowRankUpdateSymMatrixSpace::new(3, None, false);
        let mut m = space.make_new_low_rank();
        m.set_diag(dvec(&[1.0, 1.0, 1.0]) as Rc<dyn Vector>);
        // Pure diag (V, U unset) — M = I.
        let x = dvec_box(&[1.0, 2.0, 3.0]);
        let mut y = dvec_box(&[100.0, 100.0, 100.0]);
        // y ← 2*x + 0.5*y = [2,4,6] + [50,50,50] = [52, 54, 56]
        m.mult_vector(2.0, x.as_dyn_vector(), 0.5, y.as_mut());
        assert_eq!(y.expanded_values(), vec![52.0, 54.0, 56.0]);
    }

    #[test]
    fn has_valid_numbers_checks_d_v_u() {
        let space = LowRankUpdateSymMatrixSpace::new(2, None, false);
        let mut m = space.make_new_low_rank();
        m.set_diag(dvec(&[f64::NAN, 1.0]) as Rc<dyn Vector>);
        assert!(!m.has_valid_numbers());
        let mut m2 = space.make_new_low_rank();
        m2.set_diag(dvec(&[1.0, 2.0]) as Rc<dyn Vector>);
        m2.set_v(build_mv(&[&[1.0, f64::NAN]]));
        assert!(!m2.has_valid_numbers());
    }

    #[test]
    fn dim_matches_space() {
        let space = LowRankUpdateSymMatrixSpace::new(7, None, false);
        let m = space.make_new_low_rank();
        assert_eq!(m.n_rows(), 7);
        assert_eq!(m.n_cols(), 7);
        assert_eq!(m.dim(), 7);
    }
}
