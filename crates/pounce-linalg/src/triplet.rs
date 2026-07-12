//! Triplet-format matrix storage — port of
//! `LinAlg/TMatrices/IpGenTMatrix.{hpp,cpp}` and
//! `IpSymTMatrix.{hpp,cpp}`.
//!
//! Triplet (COO) format stores nonzeros as three parallel arrays:
//! `irows[k]`, `jcols[k]`, `values[k]`. **Indices are 1-based**,
//! matching upstream and the HSL convention — this is preserved
//! verbatim so the same arrays can be passed to MUMPS without
//! conversion (Phase 4). Repeated `(irow, jcol)` entries are summed at
//! the consumer (e.g. by the `TripletToCSRConverter`).
//!
//! The sparsity pattern is fixed at construction (in the matrix
//! "space"). Only the values change between calls to `set_values`.
//! `SymTMatrix` stores only one of each pair (i.e. either upper or
//! lower triangle, never both); `MultVector` walks each entry once and
//! fans the off-diagonals into both row and column of the output.

use crate::compound_vector::CompoundVector;
use crate::dense_vector::DenseVector;
use crate::matrix::{
    Matrix, MatrixCache, SymMatrix, sym_default_compute_col_amax_impl,
    sym_default_trans_mult_vector_impl,
};
use crate::vector::Vector;
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::rc::Rc;

// ---------- General triplet matrix ----------

/// Sparsity structure (pattern only) for a `GenTMatrix`. Indices are
/// 1-based.
#[derive(Debug)]
pub struct GenTMatrixSpace {
    n_rows: Index,
    n_cols: Index,
    irows: Vec<Index>,
    jcols: Vec<Index>,
}

impl GenTMatrixSpace {
    pub fn new(n_rows: Index, n_cols: Index, irows: Vec<Index>, jcols: Vec<Index>) -> Rc<Self> {
        assert_eq!(
            irows.len(),
            jcols.len(),
            "GenTMatrixSpace: irows/jcols length mismatch"
        );
        Rc::new(Self {
            n_rows,
            n_cols,
            irows,
            jcols,
        })
    }

    pub fn n_rows(&self) -> Index {
        self.n_rows
    }
    pub fn n_cols(&self) -> Index {
        self.n_cols
    }
    pub fn nonzeros(&self) -> Index {
        self.irows.len() as Index
    }
    pub fn irows(&self) -> &[Index] {
        &self.irows
    }
    pub fn jcols(&self) -> &[Index] {
        &self.jcols
    }
}

#[derive(Debug)]
pub struct GenTMatrix {
    space: Rc<GenTMatrixSpace>,
    values: Vec<Number>,
    initialized: bool,
    cache: MatrixCache,
}

impl GenTMatrix {
    pub fn new(space: Rc<GenTMatrixSpace>) -> Self {
        let nz = space.nonzeros() as usize;
        let initialized = nz == 0;
        Self {
            values: vec![0.0; nz],
            space,
            initialized,
            cache: MatrixCache::new(),
        }
    }

    pub fn space(&self) -> &Rc<GenTMatrixSpace> {
        &self.space
    }

    pub fn nonzeros(&self) -> Index {
        self.space.nonzeros()
    }
    pub fn irows(&self) -> &[Index] {
        self.space.irows()
    }
    pub fn jcols(&self) -> &[Index] {
        self.space.jcols()
    }
    pub fn values(&self) -> &[Number] {
        debug_assert!(self.initialized);
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Number] {
        // Upstream non-const Values() bumps the change tag and flips the
        // initialized flag.
        self.cache.bump();
        self.initialized = true;
        &mut self.values
    }

    pub fn set_values(&mut self, values: &[Number]) {
        assert_eq!(values.len(), self.values.len());
        self.values.copy_from_slice(values);
        self.initialized = true;
        self.cache.bump();
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl TaggedObject for GenTMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

fn downcast_dense<'a>(v: &'a dyn Vector, what: &str) -> &'a DenseVector {
    match v.as_any().downcast_ref::<DenseVector>() {
        Some(d) => d,
        None => panic!("{what} requires a DenseVector argument"),
    }
}

fn downcast_dense_mut<'a>(v: &'a mut dyn Vector, what: &str) -> &'a mut DenseVector {
    match v.as_any_mut().downcast_mut::<DenseVector>() {
        Some(d) => d,
        None => panic!("{what} requires a DenseVector argument"),
    }
}

/// Read any [`Vector`] that is either a [`DenseVector`] or a
/// [`CompoundVector`] of [`DenseVector`]s into a contiguous owned
/// `Vec<Number>`. Used by the resto-side flat-triplet matrices so they
/// can multiply against a 5-block resto-x [`CompoundVector`] without
/// each call site having to flatten manually. Panics on unsupported
/// vector kinds.
fn read_flat(v: &dyn Vector) -> Vec<Number> {
    if let Some(d) = v.as_any().downcast_ref::<DenseVector>() {
        return d.expanded_values();
    }
    if let Some(cv) = v.as_any().downcast_ref::<CompoundVector>() {
        let mut out = Vec::with_capacity(cv.dim() as usize);
        for k in 0..cv.n_comps() {
            let blk = cv.comp(k);
            let dblk = blk
                .as_any()
                .downcast_ref::<DenseVector>()
                .expect("read_flat: CompoundVector blocks must be DenseVectors in v0.1");
            out.extend_from_slice(&dblk.expanded_values());
        }
        return out;
    }
    panic!(
        "read_flat: unsupported Vector kind (need DenseVector or CompoundVector of DenseVectors)"
    );
}

/// Inverse of [`read_flat`]: write a flat slice back into a Vector
/// that is either dense or a compound of dense blocks.
fn write_flat(v: &mut dyn Vector, src: &[Number]) {
    if let Some(d) = v.as_any_mut().downcast_mut::<DenseVector>() {
        d.set_values(src);
        return;
    }
    if let Some(cv) = v.as_any_mut().downcast_mut::<CompoundVector>() {
        let mut off = 0usize;
        for k in 0..cv.n_comps() {
            let blk = cv.comp_mut(k);
            let dim = blk.dim() as usize;
            let dblk = blk
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .expect("write_flat: CompoundVector blocks must be DenseVectors in v0.1");
            dblk.set_values(&src[off..off + dim]);
            off += dim;
        }
        return;
    }
    panic!(
        "write_flat: unsupported Vector kind (need DenseVector or CompoundVector of DenseVectors)"
    );
}

impl Matrix for GenTMatrix {
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
        debug_assert!(self.initialized);
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        if self.nonzeros() == 0 {
            return;
        }
        // Resto v0.1: x or y may be a CompoundVector. Flatten on
        // entry / unflatten on exit so the inner triplet loop runs
        // against contiguous arrays.
        if x.as_any().downcast_ref::<DenseVector>().is_some()
            && y.as_any().downcast_ref::<DenseVector>().is_some()
        {
            let dx = downcast_dense(x, "GenTMatrix::mult_vector x");
            let dy = downcast_dense_mut(y, "GenTMatrix::mult_vector y");
            dy.ensure_storage();
            let irows = self.irows();
            let jcols = self.jcols();
            let yvals = dy.values_mut();
            if dx.is_homogeneous() {
                let as_ = alpha * dx.scalar();
                for (&irow, &val) in irows.iter().zip(self.values.iter()) {
                    yvals[(irow - 1) as usize] += as_ * val;
                }
            } else {
                let xvals = dx.values();
                for ((&irow, &jcol), &val) in irows.iter().zip(jcols.iter()).zip(self.values.iter())
                {
                    yvals[(irow - 1) as usize] += alpha * val * xvals[(jcol - 1) as usize];
                }
            }
            return;
        }
        let xvals = read_flat(x);
        let mut yvals = read_flat(y);
        let irows = self.irows();
        let jcols = self.jcols();
        for ((&irow, &jcol), &val) in irows.iter().zip(jcols.iter()).zip(self.values.iter()) {
            yvals[(irow - 1) as usize] += alpha * val * xvals[(jcol - 1) as usize];
        }
        write_flat(y, &yvals);
    }

    fn trans_mult_vector_impl(
        &self,
        alpha: Number,
        x: &dyn Vector,
        beta: Number,
        y: &mut dyn Vector,
    ) {
        debug_assert!(self.initialized);
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        if self.nonzeros() == 0 {
            return;
        }
        if x.as_any().downcast_ref::<DenseVector>().is_some()
            && y.as_any().downcast_ref::<DenseVector>().is_some()
        {
            let dx = downcast_dense(x, "GenTMatrix::trans_mult_vector x");
            let dy = downcast_dense_mut(y, "GenTMatrix::trans_mult_vector y");
            dy.ensure_storage();
            let irows = self.irows();
            let jcols = self.jcols();
            let yvals = dy.values_mut();
            if dx.is_homogeneous() {
                let as_ = alpha * dx.scalar();
                for (&jcol, &val) in jcols.iter().zip(self.values.iter()) {
                    yvals[(jcol - 1) as usize] += as_ * val;
                }
            } else {
                let xvals = dx.values();
                for ((&irow, &jcol), &val) in irows.iter().zip(jcols.iter()).zip(self.values.iter())
                {
                    yvals[(jcol - 1) as usize] += alpha * val * xvals[(irow - 1) as usize];
                }
            }
            return;
        }
        let xvals = read_flat(x);
        let mut yvals = read_flat(y);
        let irows = self.irows();
        let jcols = self.jcols();
        for ((&irow, &jcol), &val) in irows.iter().zip(jcols.iter()).zip(self.values.iter()) {
            yvals[(jcol - 1) as usize] += alpha * val * xvals[(irow - 1) as usize];
        }
        write_flat(y, &yvals);
    }

    fn has_valid_numbers_impl(&self) -> bool {
        debug_assert!(self.initialized);
        // Match upstream: sum the absolute values via BLAS asum and
        // check finiteness.
        let s: Number = self.values.iter().map(|v| v.abs()).sum();
        s.is_finite()
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        debug_assert!(self.initialized);
        if self.n_rows() == 0 {
            return;
        }
        let dv = downcast_dense_mut(rows_norms, "GenTMatrix::compute_row_amax");
        dv.ensure_storage();
        let irows = self.irows();
        let vec_vals = dv.values_mut();
        for (&irow, &val) in irows.iter().zip(self.values.iter()) {
            let i = (irow - 1) as usize;
            let f = val.abs();
            if f > vec_vals[i] {
                vec_vals[i] = f;
            }
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, _init: bool) {
        debug_assert!(self.initialized);
        if self.n_cols() == 0 {
            return;
        }
        let dv = downcast_dense_mut(cols_norms, "GenTMatrix::compute_col_amax");
        dv.ensure_storage();
        let jcols = self.jcols();
        let vec_vals = dv.values_mut();
        for (&jcol, &val) in jcols.iter().zip(self.values.iter()) {
            let j = (jcol - 1) as usize;
            let f = val.abs();
            if f > vec_vals[j] {
                vec_vals[j] = f;
            }
        }
    }
}

// ---------- Symmetric triplet matrix ----------

/// Sparsity structure for a symmetric triplet matrix. Stores each
/// pair only once; `irow != jcol` entries fan into both rows and
/// columns of the output during `mult_vector`.
#[derive(Debug)]
pub struct SymTMatrixSpace {
    dim: Index,
    irows: Vec<Index>,
    jcols: Vec<Index>,
}

impl SymTMatrixSpace {
    pub fn new(dim: Index, irows: Vec<Index>, jcols: Vec<Index>) -> Rc<Self> {
        assert_eq!(
            irows.len(),
            jcols.len(),
            "SymTMatrixSpace: irows/jcols length mismatch"
        );
        Rc::new(Self { dim, irows, jcols })
    }

    pub fn dim(&self) -> Index {
        self.dim
    }
    pub fn nonzeros(&self) -> Index {
        self.irows.len() as Index
    }
    pub fn irows(&self) -> &[Index] {
        &self.irows
    }
    pub fn jcols(&self) -> &[Index] {
        &self.jcols
    }
}

#[derive(Debug)]
pub struct SymTMatrix {
    space: Rc<SymTMatrixSpace>,
    values: Vec<Number>,
    initialized: bool,
    cache: MatrixCache,
}

impl SymTMatrix {
    pub fn new(space: Rc<SymTMatrixSpace>) -> Self {
        let nz = space.nonzeros() as usize;
        let initialized = nz == 0;
        Self {
            values: vec![0.0; nz],
            space,
            initialized,
            cache: MatrixCache::new(),
        }
    }

    pub fn space(&self) -> &Rc<SymTMatrixSpace> {
        &self.space
    }
    pub fn nonzeros(&self) -> Index {
        self.space.nonzeros()
    }
    pub fn irows(&self) -> &[Index] {
        self.space.irows()
    }
    pub fn jcols(&self) -> &[Index] {
        self.space.jcols()
    }
    pub fn values(&self) -> &[Number] {
        debug_assert!(self.initialized);
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Number] {
        self.cache.bump();
        self.initialized = true;
        &mut self.values
    }

    pub fn set_values(&mut self, values: &[Number]) {
        assert_eq!(values.len(), self.values.len());
        self.values.copy_from_slice(values);
        self.initialized = true;
        self.cache.bump();
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn fill_struct(&self, irn: &mut [Index], jcn: &mut [Index]) {
        debug_assert!(self.initialized);
        let nz = self.nonzeros() as usize;
        irn[..nz].copy_from_slice(self.irows());
        jcn[..nz].copy_from_slice(self.jcols());
    }

    pub fn fill_values(&self, out: &mut [Number]) {
        debug_assert!(self.initialized);
        let nz = self.nonzeros() as usize;
        out[..nz].copy_from_slice(&self.values);
    }
}

impl TaggedObject for SymTMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for SymTMatrix {
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
        debug_assert!(self.initialized);
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }
        if self.nonzeros() == 0 {
            return;
        }
        if x.as_any().downcast_ref::<DenseVector>().is_some()
            && y.as_any().downcast_ref::<DenseVector>().is_some()
        {
            let dx = downcast_dense(x, "SymTMatrix::mult_vector x");
            let dy = downcast_dense_mut(y, "SymTMatrix::mult_vector y");
            dy.ensure_storage();
            let irn = self.irows();
            let jcn = self.jcols();
            let yvals = dy.values_mut();
            if dx.is_homogeneous() {
                let as_ = alpha * dx.scalar();
                for ((&i_one, &j_one), &val) in irn.iter().zip(jcn.iter()).zip(self.values.iter()) {
                    let i = (i_one - 1) as usize;
                    let j = (j_one - 1) as usize;
                    yvals[i] += as_ * val;
                    if i_one != j_one {
                        yvals[j] += as_ * val;
                    }
                }
            } else {
                let xvals = dx.values();
                for ((&i_one, &j_one), &val) in irn.iter().zip(jcn.iter()).zip(self.values.iter()) {
                    let i = (i_one - 1) as usize;
                    let j = (j_one - 1) as usize;
                    yvals[i] += alpha * val * xvals[j];
                    if i_one != j_one {
                        yvals[j] += alpha * val * xvals[i];
                    }
                }
            }
            return;
        }
        // Resto v0.1: x or y is a CompoundVector. Flatten / unflatten.
        let xvals = read_flat(x);
        let mut yvals = read_flat(y);
        let irn = self.irows();
        let jcn = self.jcols();
        for ((&i_one, &j_one), &val) in irn.iter().zip(jcn.iter()).zip(self.values.iter()) {
            let i = (i_one - 1) as usize;
            let j = (j_one - 1) as usize;
            yvals[i] += alpha * val * xvals[j];
            if i_one != j_one {
                yvals[j] += alpha * val * xvals[i];
            }
        }
        write_flat(y, &yvals);
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
        debug_assert!(self.initialized);
        let s: Number = self.values.iter().map(|v| v.abs()).sum();
        s.is_finite()
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        debug_assert!(self.initialized);
        if self.n_rows() == 0 {
            return;
        }
        let dv = downcast_dense_mut(rows_norms, "SymTMatrix::compute_row_amax");
        dv.ensure_storage();
        // Upstream forces a zero-init regardless of the `init` flag
        // (see IpSymTMatrix.cpp:185-186). We do the same.
        let dim = self.n_rows() as usize;
        let irn = self.irows();
        let jcn = self.jcols();
        let vec_vals = dv.values_mut();
        vec_vals[..dim].fill(0.0);
        for ((&i_one, &j_one), &val) in irn.iter().zip(jcn.iter()).zip(self.values.iter()) {
            let i = (i_one - 1) as usize;
            let j = (j_one - 1) as usize;
            let f = val.abs();
            if f > vec_vals[i] {
                vec_vals[i] = f;
            }
            if f > vec_vals[j] {
                vec_vals[j] = f;
            }
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for SymTMatrix {
    fn dim(&self) -> Index {
        self.space.dim
    }
}

// ---------- TripletHelper ----------
//
// `IpTripletHelper` in upstream is a small set of free functions that
// pulls structure / values out of any Matrix into flat triplet arrays,
// and pushes flat values back in. It dispatches on RTTI to specific
// matrix kinds. We expose a `TripletHelper` namespace mirroring the
// most-used routines: `nnz`, `fill_row_col`, `fill_values`,
// `put_values_in_vector`. Full upstream coverage (CompoundMatrix
// flattening, ScaledMatrix forwarding) lands as we wire each kind into
// the IPM; the routines here cover the cases tested in this phase.

pub mod helper {
    use super::*;

    /// Number of triplet entries needed to represent `m`. For
    /// `GenTMatrix` and `SymTMatrix`, this is the matrix's `nonzeros`.
    pub fn nnz(m: &dyn Matrix) -> Index {
        if let Some(g) = m.as_any().downcast_ref::<GenTMatrix>() {
            return g.nonzeros();
        }
        if let Some(s) = m.as_any().downcast_ref::<SymTMatrix>() {
            return s.nonzeros();
        }
        panic!("TripletHelper::nnz: matrix kind not yet supported");
    }

    /// Write the triplet structure (row + col indices, 1-based) of `m`
    /// into the provided slices. Panics if a slice is shorter than
    /// `nnz(m)`.
    pub fn fill_row_col(m: &dyn Matrix, irow: &mut [Index], jcol: &mut [Index]) {
        if let Some(g) = m.as_any().downcast_ref::<GenTMatrix>() {
            let n = g.nonzeros() as usize;
            irow[..n].copy_from_slice(g.irows());
            jcol[..n].copy_from_slice(g.jcols());
            return;
        }
        if let Some(s) = m.as_any().downcast_ref::<SymTMatrix>() {
            let n = s.nonzeros() as usize;
            irow[..n].copy_from_slice(s.irows());
            jcol[..n].copy_from_slice(s.jcols());
            return;
        }
        panic!("TripletHelper::fill_row_col: matrix kind not yet supported");
    }

    /// Write the triplet values of `m` into `out`. Order matches
    /// `fill_row_col` for the same matrix.
    pub fn fill_values(m: &dyn Matrix, out: &mut [Number]) {
        if let Some(g) = m.as_any().downcast_ref::<GenTMatrix>() {
            let n = g.nonzeros() as usize;
            out[..n].copy_from_slice(g.values());
            return;
        }
        if let Some(s) = m.as_any().downcast_ref::<SymTMatrix>() {
            let n = s.nonzeros() as usize;
            out[..n].copy_from_slice(s.values());
            return;
        }
        panic!("TripletHelper::fill_values: matrix kind not yet supported");
    }

    /// Copy a flat dense slice into a `DenseVector`. Mirror of upstream
    /// `TripletHelper::PutValuesInVector(Index dim, const Number* vals,
    /// Vector& v)` — only the DenseVector specialisation is needed for
    /// Phase 2 tests.
    pub fn put_values_in_dense_vector(values: &[Number], v: &mut dyn Vector) {
        let dv = match v.as_any_mut().downcast_mut::<DenseVector>() {
            Some(d) => d,
            None => panic!("TripletHelper::put_values_in_dense_vector requires a DenseVector"),
        };
        dv.set_values(values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_vector::DenseVectorSpace;

    fn dvec_box(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    #[test]
    fn gen_t_matrix_mult_vector() {
        // 2x3 matrix:
        //  [ 1 0 2 ]
        //  [ 0 3 4 ]
        // Triplets (1-based): (1,1)=1, (1,3)=2, (2,2)=3, (2,3)=4.
        let space = GenTMatrixSpace::new(2, 3, vec![1, 1, 2, 2], vec![1, 3, 2, 3]);
        let mut m = GenTMatrix::new(space);
        m.set_values(&[1.0, 2.0, 3.0, 4.0]);

        let x = dvec_box(&[5.0, 7.0, 11.0]);
        let mut y = dvec_box(&[0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        // [1*5 + 2*11, 3*7 + 4*11] = [27, 65]
        let dy = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dy.values(), &[27.0, 65.0]);
    }

    #[test]
    fn gen_t_matrix_trans_mult_vector() {
        // Same matrix as above. M^T * [5, 7] = [5, 21, 10 + 28] = [5, 21, 38]
        let space = GenTMatrixSpace::new(2, 3, vec![1, 1, 2, 2], vec![1, 3, 2, 3]);
        let mut m = GenTMatrix::new(space);
        m.set_values(&[1.0, 2.0, 3.0, 4.0]);

        let x = dvec_box(&[5.0, 7.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        m.trans_mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        let dy = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dy.values(), &[5.0, 21.0, 38.0]);
    }

    #[test]
    fn gen_t_matrix_compute_amax() {
        // Use signed values so abs matters.
        let space = GenTMatrixSpace::new(2, 3, vec![1, 1, 2, 2], vec![1, 3, 2, 3]);
        let mut m = GenTMatrix::new(space);
        m.set_values(&[-1.0, 2.0, -3.0, 4.0]);

        let mut row_norms = dvec_box(&[0.0, 0.0]);
        m.compute_row_amax(row_norms.as_mut(), true);
        let dv = row_norms.as_any().downcast_ref::<DenseVector>().unwrap();
        // Row 1: max(|-1|,|2|) = 2; Row 2: max(|-3|,|4|) = 4.
        assert_eq!(dv.values(), &[2.0, 4.0]);

        let mut col_norms = dvec_box(&[0.0, 0.0, 0.0]);
        m.compute_col_amax(col_norms.as_mut(), true);
        let dv = col_norms.as_any().downcast_ref::<DenseVector>().unwrap();
        // Col 1: 1; Col 2: 3; Col 3: max(2, 4) = 4.
        assert_eq!(dv.values(), &[1.0, 3.0, 4.0]);
    }

    #[test]
    fn gen_t_matrix_has_valid_numbers_detects_nan() {
        let space = GenTMatrixSpace::new(2, 2, vec![1, 2], vec![1, 2]);
        let mut m = GenTMatrix::new(space);
        m.set_values(&[1.0, f64::NAN]);
        assert!(!m.has_valid_numbers());
    }

    #[test]
    fn sym_t_matrix_mult_vector() {
        // 3x3 sym:
        //  [ 1 2 0 ]
        //  [ 2 4 5 ]
        //  [ 0 5 6 ]
        // Triplets (1-based, store one of each pair):
        //   (1,1)=1, (2,1)=2, (2,2)=4, (3,2)=5, (3,3)=6
        let space = SymTMatrixSpace::new(3, vec![1, 2, 2, 3, 3], vec![1, 1, 2, 2, 3]);
        let mut m = SymTMatrix::new(space);
        m.set_values(&[1.0, 2.0, 4.0, 5.0, 6.0]);

        let x = dvec_box(&[1.0, 1.0, 1.0]);
        let mut y = dvec_box(&[0.0, 0.0, 0.0]);
        m.mult_vector(1.0, x.as_dyn_vector(), 0.0, y.as_mut());
        let dy = y.as_any().downcast_ref::<DenseVector>().unwrap();
        // M * [1,1,1] = [3, 11, 11]
        assert_eq!(dy.values(), &[3.0, 11.0, 11.0]);
    }

    #[test]
    fn sym_t_matrix_homogeneous_x_path() {
        let space = SymTMatrixSpace::new(2, vec![1, 2, 2], vec![1, 1, 2]);
        let mut m = SymTMatrix::new(space);
        m.set_values(&[1.0, 2.0, 3.0]);
        // M = [[1,2],[2,3]]. M * [k,k] = [3k, 5k]; for k=2 -> [6, 10].
        let space2 = DenseVectorSpace::new(2);
        let mut x = space2.make_new_dense();
        x.set(2.0); // homogeneous
        let mut y = dvec_box(&[0.0, 0.0]);
        m.mult_vector(1.0, &x, 0.0, y.as_mut());
        let dy = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dy.values(), &[6.0, 10.0]);
    }

    #[test]
    fn sym_t_matrix_compute_row_amax_includes_off_diagonal() {
        let space = SymTMatrixSpace::new(2, vec![1, 2, 2], vec![1, 1, 2]);
        let mut m = SymTMatrix::new(space);
        m.set_values(&[1.0, -7.0, 3.0]);
        // M = [[1,-7],[-7,3]]. Row 0 max = 7, Row 1 max = 7.
        let mut nrm = dvec_box(&[0.0, 0.0]);
        m.compute_row_amax(nrm.as_mut(), true);
        let dv = nrm.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dv.values(), &[7.0, 7.0]);
    }

    #[test]
    fn triplet_helper_round_trip_gen() {
        let space = GenTMatrixSpace::new(2, 2, vec![1, 2], vec![1, 2]);
        let mut m = GenTMatrix::new(space);
        m.set_values(&[5.0, 9.0]);
        assert_eq!(helper::nnz(&m), 2);
        let mut irow = vec![0i32; 2];
        let mut jcol = vec![0i32; 2];
        helper::fill_row_col(&m, &mut irow, &mut jcol);
        assert_eq!(irow, vec![1, 2]);
        assert_eq!(jcol, vec![1, 2]);
        let mut vals = vec![0.0; 2];
        helper::fill_values(&m, &mut vals);
        assert_eq!(vals, vec![5.0, 9.0]);
    }
}
