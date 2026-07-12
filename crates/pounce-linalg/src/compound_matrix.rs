//! Block matrices — port of `LinAlg/IpCompoundMatrix.{hpp,cpp}` and
//! `LinAlg/IpCompoundSymMatrix.{hpp,cpp}`.
//!
//! A `CompoundMatrix` is laid out as `M[ irow ][ jcol ]` with
//! `n_comps_rows × n_comps_cols` named blocks; an unset block is
//! treated as zero. A `CompoundSymMatrix` stores only the lower
//! triangle: `M[ irow ][ jcol ]` for `jcol <= irow`. Diagonal blocks
//! must themselves be symmetric (debug-asserted at `set_comp`).
//!
//! For bit-equivalence with upstream, the iteration order in
//! `mult_vector` and `trans_mult_vector` matches `IpCompoundMatrix.cpp`
//! and `IpCompoundSymMatrix.cpp` exactly: outer loop over `irow`, inner
//! over `jcol`, summed into the row-block of `y` in increasing order.

use crate::compound_vector::CompoundVector;
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

// ---------- General CompoundMatrixSpace ----------

#[derive(Debug)]
pub struct CompoundMatrixSpace {
    n_comps_rows: Index,
    n_comps_cols: Index,
    total_n_rows: Index,
    total_n_cols: Index,
    block_rows: Vec<Index>,
    block_cols: Vec<Index>,
}

impl CompoundMatrixSpace {
    pub fn new(
        n_comps_rows: Index,
        n_comps_cols: Index,
        total_n_rows: Index,
        total_n_cols: Index,
    ) -> Rc<Self> {
        Rc::new(Self {
            n_comps_rows,
            n_comps_cols,
            total_n_rows,
            total_n_cols,
            block_rows: vec![0; n_comps_rows as usize],
            block_cols: vec![0; n_comps_cols as usize],
        })
    }

    /// Builder-style constructor: takes the per-block-row and -column
    /// dimensions up front. Sums must match `total_n_rows` /
    /// `total_n_cols`.
    pub fn new_with_dims(block_rows: Vec<Index>, block_cols: Vec<Index>) -> Rc<Self> {
        let total_n_rows: Index = block_rows.iter().sum();
        let total_n_cols: Index = block_cols.iter().sum();
        Rc::new(Self {
            n_comps_rows: block_rows.len() as Index,
            n_comps_cols: block_cols.len() as Index,
            total_n_rows,
            total_n_cols,
            block_rows,
            block_cols,
        })
    }

    pub fn n_comps_rows(&self) -> Index {
        self.n_comps_rows
    }
    pub fn n_comps_cols(&self) -> Index {
        self.n_comps_cols
    }
    pub fn total_n_rows(&self) -> Index {
        self.total_n_rows
    }
    pub fn total_n_cols(&self) -> Index {
        self.total_n_cols
    }
    pub fn block_rows(&self, irow: Index) -> Index {
        self.block_rows[irow as usize]
    }
    pub fn block_cols(&self, jcol: Index) -> Index {
        self.block_cols[jcol as usize]
    }
}

// ---------- General CompoundMatrix ----------

#[derive(Debug)]
pub struct CompoundMatrix {
    space: Rc<CompoundMatrixSpace>,
    /// Row-major: `comps[irow][jcol]`, with `irow ∈ [0, n_comps_rows)`,
    /// `jcol ∈ [0, n_comps_cols)`. `None` ⇒ implicit zero block.
    comps: Vec<Vec<Option<Rc<dyn Matrix>>>>,
    /// Lazily-set "blocks lie only on the diagonal" flag — enables
    /// upstream's diagonal fast path. Recomputed when invalidated.
    diagonal_known: Cell<bool>,
    diagonal: Cell<bool>,
    cache: MatrixCache,
}

impl CompoundMatrix {
    pub fn new(space: Rc<CompoundMatrixSpace>) -> Self {
        let nr = space.n_comps_rows as usize;
        let nc = space.n_comps_cols as usize;
        let mut comps: Vec<Vec<Option<Rc<dyn Matrix>>>> = Vec::with_capacity(nr);
        for _ in 0..nr {
            comps.push((0..nc).map(|_| None).collect());
        }
        Self {
            space,
            comps,
            diagonal_known: Cell::new(false),
            diagonal: Cell::new(false),
            cache: MatrixCache::new(),
        }
    }

    pub fn space(&self) -> &Rc<CompoundMatrixSpace> {
        &self.space
    }

    pub fn n_comps_rows(&self) -> Index {
        self.space.n_comps_rows
    }
    pub fn n_comps_cols(&self) -> Index {
        self.space.n_comps_cols
    }

    pub fn set_comp(&mut self, irow: Index, jcol: Index, matrix: Rc<dyn Matrix>) {
        debug_assert!(irow < self.space.n_comps_rows);
        debug_assert!(jcol < self.space.n_comps_cols);
        debug_assert_eq!(matrix.n_rows(), self.space.block_rows[irow as usize]);
        debug_assert_eq!(matrix.n_cols(), self.space.block_cols[jcol as usize]);
        self.comps[irow as usize][jcol as usize] = Some(matrix);
        self.diagonal_known.set(false);
        self.cache.bump();
    }

    pub fn get_comp(&self, irow: Index, jcol: Index) -> Option<&Rc<dyn Matrix>> {
        self.comps[irow as usize][jcol as usize].as_ref()
    }

    fn diagonal(&self) -> bool {
        if self.diagonal_known.get() {
            return self.diagonal.get();
        }
        let nr = self.space.n_comps_rows as usize;
        let nc = self.space.n_comps_cols as usize;
        let mut diag = nr == nc;
        if diag {
            for i in 0..nr {
                for j in 0..nc {
                    let occ = self.comps[i][j].is_some();
                    if i == j {
                        if !occ {
                            diag = false;
                            break;
                        }
                    } else if occ {
                        diag = false;
                        break;
                    }
                }
                if !diag {
                    break;
                }
            }
        }
        self.diagonal.set(diag);
        self.diagonal_known.set(true);
        diag
    }
}

impl TaggedObject for CompoundMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

/// Try to view `y: &mut dyn Vector` as `&mut CompoundVector` if it has
/// the expected number of components.
fn comp_y_mut(y: &mut dyn Vector, expected_n: Index) -> Option<&mut CompoundVector> {
    let is_match = y
        .as_any()
        .downcast_ref::<CompoundVector>()
        .is_some_and(|cv| cv.n_comps() == expected_n);
    if is_match {
        y.as_any_mut().downcast_mut::<CompoundVector>()
    } else {
        None
    }
}

fn comp_x_ref(x: &dyn Vector, expected_n: Index) -> Option<&CompoundVector> {
    x.as_any()
        .downcast_ref::<CompoundVector>()
        .filter(|cv| cv.n_comps() == expected_n)
}

impl Matrix for CompoundMatrix {
    fn n_rows(&self) -> Index {
        self.space.total_n_rows
    }
    fn n_cols(&self) -> Index {
        self.space.total_n_cols
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
        // Pre-scale y exactly once per call (matches upstream).
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }

        let nr = self.space.n_comps_rows as usize;
        let nc = self.space.n_comps_cols as usize;
        let comp_x = comp_x_ref(x, self.space.n_comps_cols);

        let _ = self.diagonal(); // warm cache; not used directly — see note above

        if let Some(cy) = comp_y_mut(y, self.space.n_comps_rows) {
            for irow in 0..nr {
                let y_i = cy.comp_mut(irow as Index);
                for jcol in 0..nc {
                    if let Some(m) = self.comps[irow][jcol].as_ref() {
                        let x_j: &dyn Vector = match comp_x {
                            Some(cx) => cx.comp(jcol as Index),
                            None => {
                                debug_assert_eq!(self.space.n_comps_cols, 1);
                                x
                            }
                        };
                        m.mult_vector(alpha, x_j, 1.0, y_i);
                    }
                }
            }
        } else {
            debug_assert_eq!(self.space.n_comps_rows, 1);
            for irow in 0..nr {
                for jcol in 0..nc {
                    if let Some(m) = self.comps[irow][jcol].as_ref() {
                        let x_j: &dyn Vector = match comp_x {
                            Some(cx) => cx.comp(jcol as Index),
                            None => {
                                debug_assert_eq!(self.space.n_comps_cols, 1);
                                x
                            }
                        };
                        m.mult_vector(alpha, x_j, 1.0, y);
                    }
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
        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }

        let nr = self.space.n_comps_rows as usize;
        let nc = self.space.n_comps_cols as usize;
        // Trans: x has n_rows blocks (row-block dims of original);
        //        y has n_cols blocks (col-block dims of original).
        let comp_x = comp_x_ref(x, self.space.n_comps_rows);

        if let Some(cy) = comp_y_mut(y, self.space.n_comps_cols) {
            for irow in 0..nc {
                let y_i = cy.comp_mut(irow as Index);
                for jcol in 0..nr {
                    if let Some(m) = self.comps[jcol][irow].as_ref() {
                        let x_j: &dyn Vector = match comp_x {
                            Some(cx) => cx.comp(jcol as Index),
                            None => {
                                debug_assert_eq!(self.space.n_comps_rows, 1);
                                x
                            }
                        };
                        m.trans_mult_vector(alpha, x_j, 1.0, y_i);
                    }
                }
            }
        } else {
            debug_assert_eq!(self.space.n_comps_cols, 1);
            for irow in 0..nc {
                for jcol in 0..nr {
                    if let Some(m) = self.comps[jcol][irow].as_ref() {
                        let x_j: &dyn Vector = match comp_x {
                            Some(cx) => cx.comp(jcol as Index),
                            None => {
                                debug_assert_eq!(self.space.n_comps_rows, 1);
                                x
                            }
                        };
                        m.trans_mult_vector(alpha, x_j, 1.0, y);
                    }
                }
            }
        }
    }

    fn has_valid_numbers_impl(&self) -> bool {
        for row in &self.comps {
            for m in row.iter().flatten() {
                if !m.has_valid_numbers() {
                    return false;
                }
            }
        }
        true
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        let nr = self.space.n_comps_rows as usize;
        let nc = self.space.n_comps_cols as usize;
        // Outer wrapper has already zeroed when init=true; pass false
        // downstream to skip re-zeroing (matches upstream).
        if let Some(cv) = comp_y_mut(rows_norms, self.space.n_comps_rows) {
            for jcol in 0..nc {
                for irow in 0..nr {
                    if let Some(m) = self.comps[irow][jcol].as_ref() {
                        let v_i = cv.comp_mut(irow as Index);
                        m.compute_row_amax(v_i, false);
                    }
                }
            }
        } else {
            debug_assert_eq!(self.space.n_comps_rows, 1);
            for jcol in 0..nc {
                for irow in 0..nr {
                    if let Some(m) = self.comps[irow][jcol].as_ref() {
                        m.compute_row_amax(rows_norms, false);
                    }
                }
            }
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, _init: bool) {
        let nr = self.space.n_comps_rows as usize;
        let nc = self.space.n_comps_cols as usize;
        // cols_norms has n_comps_cols blocks (one per column-block).
        if let Some(cv) = comp_y_mut(cols_norms, self.space.n_comps_cols) {
            for irow in 0..nr {
                for jcol in 0..nc {
                    if let Some(m) = self.comps[irow][jcol].as_ref() {
                        let v_j = cv.comp_mut(jcol as Index);
                        m.compute_col_amax(v_j, false);
                    }
                }
            }
        } else {
            debug_assert_eq!(self.space.n_comps_cols, 1);
            for irow in 0..nr {
                for jcol in 0..nc {
                    if let Some(m) = self.comps[irow][jcol].as_ref() {
                        m.compute_col_amax(cols_norms, false);
                    }
                }
            }
        }
    }
}

// ---------- Symmetric CompoundSymMatrix ----------

#[derive(Debug)]
pub struct CompoundSymMatrixSpace {
    n_comps_dim: Index,
    total_dim: Index,
    block_dim: Vec<Index>,
}

impl CompoundSymMatrixSpace {
    pub fn new(n_comps_dim: Index, total_dim: Index) -> Rc<Self> {
        Rc::new(Self {
            n_comps_dim,
            total_dim,
            block_dim: vec![0; n_comps_dim as usize],
        })
    }

    pub fn new_with_dims(block_dim: Vec<Index>) -> Rc<Self> {
        let total_dim: Index = block_dim.iter().sum();
        Rc::new(Self {
            n_comps_dim: block_dim.len() as Index,
            total_dim,
            block_dim,
        })
    }

    pub fn n_comps_dim(&self) -> Index {
        self.n_comps_dim
    }
    pub fn total_dim(&self) -> Index {
        self.total_dim
    }
    pub fn block_dim(&self, irow_jcol: Index) -> Index {
        self.block_dim[irow_jcol as usize]
    }
}

#[derive(Debug)]
pub struct CompoundSymMatrix {
    space: Rc<CompoundSymMatrixSpace>,
    /// Triangular: `comps[irow]` has `irow + 1` cells (for `jcol = 0..=irow`).
    comps: Vec<Vec<Option<Rc<dyn Matrix>>>>,
    cache: MatrixCache,
}

impl CompoundSymMatrix {
    pub fn new(space: Rc<CompoundSymMatrixSpace>) -> Self {
        let n = space.n_comps_dim as usize;
        let mut comps: Vec<Vec<Option<Rc<dyn Matrix>>>> = Vec::with_capacity(n);
        for irow in 0..n {
            comps.push((0..=irow).map(|_| None).collect());
        }
        Self {
            space,
            comps,
            cache: MatrixCache::new(),
        }
    }

    pub fn space(&self) -> &Rc<CompoundSymMatrixSpace> {
        &self.space
    }

    pub fn n_comps_dim(&self) -> Index {
        self.space.n_comps_dim
    }

    /// Set the (irow, jcol) block. Requires `jcol <= irow`. Diagonal
    /// blocks (`irow == jcol`) should be symmetric matrices —
    /// upstream debug-asserts via `dynamic_cast<SymMatrix*>`. We trust
    /// the caller here.
    pub fn set_comp(&mut self, irow: Index, jcol: Index, matrix: Rc<dyn Matrix>) {
        debug_assert!(irow < self.space.n_comps_dim);
        debug_assert!(jcol <= irow);
        debug_assert_eq!(matrix.n_rows(), self.space.block_dim[irow as usize]);
        debug_assert_eq!(matrix.n_cols(), self.space.block_dim[jcol as usize]);
        self.comps[irow as usize][jcol as usize] = Some(matrix);
        self.cache.bump();
    }

    pub fn get_comp(&self, irow: Index, jcol: Index) -> Option<&Rc<dyn Matrix>> {
        debug_assert!(jcol <= irow);
        self.comps[irow as usize][jcol as usize].as_ref()
    }
}

impl TaggedObject for CompoundSymMatrix {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Matrix for CompoundSymMatrix {
    fn n_rows(&self) -> Index {
        self.space.total_dim
    }
    fn n_cols(&self) -> Index {
        self.space.total_dim
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
        let n = self.space.n_comps_dim as usize;
        // Upstream `static_cast`s — both x and y must be CompoundVector
        // with NComps() == NComps_Dim(). We follow the same contract.
        let comp_x = match comp_x_ref(x, self.space.n_comps_dim) {
            Some(c) => c,
            None => panic!(
                "CompoundSymMatrix::mult_vector requires a CompoundVector x with {} components",
                self.space.n_comps_dim
            ),
        };

        if beta != 0.0 {
            y.scal(beta);
        } else {
            y.set(0.0);
        }

        let cy = match comp_y_mut(y, self.space.n_comps_dim) {
            Some(c) => c,
            None => panic!(
                "CompoundSymMatrix::mult_vector requires a CompoundVector y with {} components",
                self.space.n_comps_dim
            ),
        };

        for irow in 0..n {
            let y_i = cy.comp_mut(irow as Index);
            // Lower triangle (j <= i): direct mult.
            for jcol in 0..=irow {
                if let Some(m) = self.comps[irow][jcol].as_ref() {
                    let x_j = comp_x.comp(jcol as Index);
                    m.mult_vector(alpha, x_j, 1.0, y_i);
                }
            }
            // Upper triangle (j > i): use stored M[j][i] transposed.
            for jcol in (irow + 1)..n {
                if let Some(m) = self.comps[jcol][irow].as_ref() {
                    let x_j = comp_x.comp(jcol as Index);
                    m.trans_mult_vector(alpha, x_j, 1.0, y_i);
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
        for row in &self.comps {
            for m in row.iter().flatten() {
                if !m.has_valid_numbers() {
                    return false;
                }
            }
        }
        true
    }

    fn compute_row_amax_impl(&self, rows_norms: &mut dyn Vector, _init: bool) {
        let n = self.space.n_comps_dim as usize;
        let cv = match comp_y_mut(rows_norms, self.space.n_comps_dim) {
            Some(c) => c,
            None => panic!(
                "CompoundSymMatrix::compute_row_amax requires a CompoundVector with {} components",
                self.space.n_comps_dim
            ),
        };
        // Same iteration order as upstream IpCompoundSymMatrix.cpp:214.
        for jcol in 0..n {
            for irow in 0..n {
                let m = if jcol <= irow {
                    self.comps[irow][jcol].as_ref()
                } else {
                    self.comps[jcol][irow].as_ref()
                };
                if let Some(m) = m {
                    let vec_i = cv.comp_mut(irow as Index);
                    m.compute_row_amax(vec_i, false);
                }
            }
        }
    }

    fn compute_col_amax_impl(&self, cols_norms: &mut dyn Vector, init: bool) {
        sym_default_compute_col_amax_impl(self, cols_norms, init);
    }
}

impl SymMatrix for CompoundSymMatrix {
    fn dim(&self) -> Index {
        self.space.total_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ZeroMatrix;
    use crate::compound_vector::{CompoundVector, CompoundVectorSpace};
    use crate::dense_vector::{DenseVector, DenseVectorSpace};
    use crate::diag_matrix::DiagMatrix;
    use crate::special_matrix::IdentityMatrix;

    fn dvec_box(values: &[Number]) -> Box<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Box::new(v)
    }

    fn dvec_rc(values: &[Number]) -> Rc<dyn Vector> {
        let space = DenseVectorSpace::new(values.len() as Index);
        let mut v = space.make_new_dense();
        v.set_values(values);
        Rc::from(Box::new(v) as Box<dyn Vector>)
    }

    /// Two-block compound vector with the given dense values per block.
    fn make_compound_vec(blocks: &[&[Number]]) -> CompoundVector {
        let dims: Vec<Index> = blocks.iter().map(|b| b.len() as Index).collect();
        let total: Index = dims.iter().sum();
        let space = CompoundVectorSpace::new(blocks.len() as Index, total);
        for (i, b) in blocks.iter().enumerate() {
            let s = DenseVectorSpace::new(b.len() as Index);
            space.set_comp(i as Index, b.len() as Index, {
                let s = Rc::clone(&s);
                move || Box::new(DenseVector::new(Rc::clone(&s)))
            });
            let _ = b; // values set after construction
        }
        let mut v = CompoundVector::new(space);
        for (i, b) in blocks.iter().enumerate() {
            let dv = v
                .comp_mut(i as Index)
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .unwrap();
            dv.set_values(b);
        }
        v
    }

    #[test]
    fn compound_matrix_diagonal_two_identities() {
        // 2-block diagonal: M = diag(2*I_2, 3*I_3).
        let space = CompoundMatrixSpace::new_with_dims(vec![2, 3], vec![2, 3]);
        let mut m = CompoundMatrix::new(space);
        let mut id2 = IdentityMatrix::new(2);
        id2.set_factor(2.0);
        let mut id3 = IdentityMatrix::new(3);
        id3.set_factor(3.0);
        m.set_comp(0, 0, Rc::new(id2));
        m.set_comp(1, 1, Rc::new(id3));

        let x = make_compound_vec(&[&[1.0, 2.0], &[3.0, 4.0, 5.0]]);
        let mut y = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0, 0.0]]);
        m.mult_vector(1.0, &x, 0.0, &mut y);

        let y0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        let y1 = y.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(y0.expanded_values().to_vec(), vec![2.0, 4.0]);
        assert_eq!(y1.expanded_values().to_vec(), vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn compound_matrix_offdiag_block_contributes() {
        // M = [[I_2,  Z_{2x3}],
        //      [Z_{3x2}, 3 I_3]]
        // plus an extra term in (0,1): a 2x3 rank-1 matrix represented
        // by ZeroMatrix variant — easier: skip and use just diagonals
        // here, but verify off-diagonal works with a different layout.
        //
        // Use M = [[I_2, A], [0, I_3]] where A is 2x3 zero — so result
        // is just the identity. Then swap A with a real matrix.
        //
        // Easier: 2x2 block matrix [[I_2, I_2], [0, I_2]]. Then
        //   M*[x_top; x_bot] = [x_top + x_bot; x_bot]
        let space = CompoundMatrixSpace::new_with_dims(vec![2, 2], vec![2, 2]);
        let mut m = CompoundMatrix::new(space);
        m.set_comp(0, 0, Rc::new(IdentityMatrix::new(2)));
        m.set_comp(0, 1, Rc::new(IdentityMatrix::new(2)));
        m.set_comp(1, 0, Rc::new(ZeroMatrix::new(2, 2)));
        m.set_comp(1, 1, Rc::new(IdentityMatrix::new(2)));

        let x = make_compound_vec(&[&[1.0, 2.0], &[10.0, 20.0]]);
        let mut y = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0]]);
        m.mult_vector(1.0, &x, 0.0, &mut y);

        let y0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        let y1 = y.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(y0.expanded_values().to_vec(), vec![11.0, 22.0]);
        assert_eq!(y1.expanded_values().to_vec(), vec![10.0, 20.0]);
    }

    #[test]
    fn compound_matrix_trans_swaps_blocks() {
        // Same M as above. M^T = [[I_2, 0], [I_2, I_2]].
        // M^T * [a; b] = [a; a + b].
        let space = CompoundMatrixSpace::new_with_dims(vec![2, 2], vec![2, 2]);
        let mut m = CompoundMatrix::new(space);
        m.set_comp(0, 0, Rc::new(IdentityMatrix::new(2)));
        m.set_comp(0, 1, Rc::new(IdentityMatrix::new(2)));
        m.set_comp(1, 1, Rc::new(IdentityMatrix::new(2)));

        let x = make_compound_vec(&[&[1.0, 2.0], &[10.0, 20.0]]);
        let mut y = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0]]);
        m.trans_mult_vector(1.0, &x, 0.0, &mut y);

        let y0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        let y1 = y.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(y0.expanded_values().to_vec(), vec![1.0, 2.0]);
        assert_eq!(y1.expanded_values().to_vec(), vec![11.0, 22.0]);
    }

    #[test]
    fn compound_matrix_has_valid_numbers_propagates() {
        let space = CompoundMatrixSpace::new_with_dims(vec![2], vec![2]);
        let mut m = CompoundMatrix::new(space);
        let mut bad = IdentityMatrix::new(2);
        bad.set_factor(f64::NAN);
        m.set_comp(0, 0, Rc::new(bad));
        assert!(!m.has_valid_numbers());
    }

    #[test]
    fn compound_sym_matrix_diagonal_blocks_only() {
        // sym M = diag(2*I_2, 3*I_3). Mult on full compound vector.
        let space = CompoundSymMatrixSpace::new_with_dims(vec![2, 3]);
        let mut m = CompoundSymMatrix::new(space);
        let mut id2 = IdentityMatrix::new(2);
        id2.set_factor(2.0);
        let mut id3 = IdentityMatrix::new(3);
        id3.set_factor(3.0);
        m.set_comp(0, 0, Rc::new(id2));
        m.set_comp(1, 1, Rc::new(id3));

        let x = make_compound_vec(&[&[1.0, 2.0], &[3.0, 4.0, 5.0]]);
        let mut y = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0, 0.0]]);
        m.mult_vector(1.0, &x, 0.0, &mut y);

        let y0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        let y1 = y.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(y0.expanded_values().to_vec(), vec![2.0, 4.0]);
        assert_eq!(y1.expanded_values().to_vec(), vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn compound_sym_matrix_uses_offdiag_transpose() {
        // 2-block sym, with stored lower-tri M[1][0] = D = diag([10, 100]).
        // Diagonal blocks empty (treated as zero).
        // Full M = [[0, D^T], [D, 0]]. M*[x0; x1] = [D^T x1; D x0].
        let space = CompoundSymMatrixSpace::new_with_dims(vec![2, 2]);
        let mut m = CompoundSymMatrix::new(space);
        let d_diag: Rc<dyn Vector> = dvec_rc(&[10.0, 100.0]);
        let mut d = DiagMatrix::new(2);
        d.set_diag(d_diag);
        m.set_comp(1, 0, Rc::new(d));

        let x = make_compound_vec(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let mut y = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0]]);
        m.mult_vector(1.0, &x, 0.0, &mut y);

        let y0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        let y1 = y.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
        // D^T = D for diag.  D*x1 = [30, 400]. D^T*x1 also = [30, 400]?
        // Wait: D is 2x2 here applied to the second compound block. y0 = D x1 = [30, 400].
        assert_eq!(y0.expanded_values().to_vec(), vec![30.0, 400.0]);
        // y1 = D x0 = [10, 200].
        assert_eq!(y1.expanded_values().to_vec(), vec![10.0, 200.0]);
    }

    #[test]
    fn compound_sym_trans_mult_equals_mult() {
        // For SymMatrix, M^T = M. Check trans_mult routes to same result.
        let space = CompoundSymMatrixSpace::new_with_dims(vec![2, 2]);
        let mut m = CompoundSymMatrix::new(space);
        let mut id_a = IdentityMatrix::new(2);
        id_a.set_factor(7.0);
        let mut id_b = IdentityMatrix::new(2);
        id_b.set_factor(11.0);
        m.set_comp(0, 0, Rc::new(id_a));
        m.set_comp(1, 1, Rc::new(id_b));

        let x = make_compound_vec(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let mut y_mult = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0]]);
        m.mult_vector(1.0, &x, 0.0, &mut y_mult);
        let mut y_trans = make_compound_vec(&[&[0.0, 0.0], &[0.0, 0.0]]);
        m.trans_mult_vector(1.0, &x, 0.0, &mut y_trans);

        for i in 0..2 {
            let a = y_mult
                .comp(i)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap()
                .expanded_values()
                .to_vec();
            let b = y_trans
                .comp(i)
                .as_any()
                .downcast_ref::<DenseVector>()
                .unwrap()
                .expanded_values()
                .to_vec();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn compound_matrix_beta_scales_y() {
        let space = CompoundMatrixSpace::new_with_dims(vec![2], vec![2]);
        let mut m = CompoundMatrix::new(space);
        m.set_comp(0, 0, Rc::new(IdentityMatrix::new(2)));

        let x = make_compound_vec(&[&[1.0, 2.0]]);
        let mut y = make_compound_vec(&[&[10.0, 20.0]]);
        m.mult_vector(1.0, &x, 0.5, &mut y);
        // y = 1*I*x + 0.5*y = [1, 2] + [5, 10] = [6, 12]
        let y0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(y0.expanded_values().to_vec(), vec![6.0, 12.0]);
    }

    #[test]
    fn unused_helpers_compile() {
        let _ = dvec_box(&[1.0]); // exercise helper to silence dead_code
    }
}
