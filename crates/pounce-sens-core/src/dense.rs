//! Dense linear algebra on the small matrices the statistics build: a
//! parameter block of the inverse KKT matrix, its reduced Hessian, the
//! projections onto binding rows. faer does the work. Each routine states
//! which `numpy.linalg` call it answers for, and where the answer is a
//! decision, such as a rank or a singular test, it takes numpy's tolerance.

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use pounce_common::types::Number;

/// A dense row-major matrix, the layout every caller here builds.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    /// Rows.
    pub rows: usize,
    /// Columns.
    pub cols: usize,
    /// Entries, row by row.
    pub data: Vec<Number>,
}

impl Dense {
    /// A matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// The identity.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// A matrix from a function of the row and column.
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> Number) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }

    /// Entry `(i, j)`.
    pub fn at(&self, i: usize, j: usize) -> Number {
        self.data[i * self.cols + j]
    }

    /// Sets entry `(i, j)`.
    pub fn set(&mut self, i: usize, j: usize, v: Number) {
        self.data[i * self.cols + j] = v;
    }

    /// The transpose.
    pub fn t(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |i, j| self.at(j, i))
    }

    /// `self · other`.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "dense product of mismatched shapes");
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.at(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * other.cols + j] += a * other.at(k, j);
                }
            }
        }
        out
    }

    /// `self · v`.
    pub fn mul_vec(&self, v: &[Number]) -> Vec<Number> {
        assert_eq!(self.cols, v.len(), "dense product of mismatched shapes");
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self.at(i, j) * v[j]).sum())
            .collect()
    }

    /// `self + other`.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// `self - other`.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// `s · self`.
    pub fn scale(&self, s: Number) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|a| s * a).collect(),
        }
    }

    /// `(self + selfᵀ) / 2`.
    pub fn symmetrized(&self) -> Self {
        Self::from_fn(self.rows, self.cols, |i, j| {
            0.5 * (self.at(i, j) + self.at(j, i))
        })
    }

    /// The submatrix on `rows` × `cols`, in the order given.
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> Self {
        Self::from_fn(rows.len(), cols.len(), |i, j| self.at(rows[i], cols[j]))
    }

    /// The diagonal.
    pub fn diagonal(&self) -> Vec<Number> {
        (0..self.rows.min(self.cols))
            .map(|i| self.at(i, i))
            .collect()
    }

    /// The largest absolute entry, zero for an empty matrix.
    pub fn max_abs(&self) -> Number {
        self.data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
    }

    fn to_faer(&self) -> Mat<Number> {
        Mat::from_fn(self.rows, self.cols, |i, j| self.at(i, j))
    }

    fn from_faer(m: faer::MatRef<'_, Number>) -> Self {
        Self::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)])
    }
}

/// The Euclidean norm, `numpy.linalg.norm` of a vector.
pub fn norm(v: &[Number]) -> Number {
    v.iter().map(|a| a * a).sum::<Number>().sqrt()
}

/// The dot product.
pub fn dot(a: &[Number], b: &[Number]) -> Number {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The LU factorization with partial pivoting, or `None` when a pivot is
/// exactly zero, which is when `numpy.linalg.inv` and `solve` raise
/// `LinAlgError` (LAPACK's `getrf` reports exactly that).
fn lu(a: &Dense) -> Option<faer::linalg::solvers::PartialPivLu<Number>> {
    assert_eq!(a.rows, a.cols, "a square matrix is required");
    let lu = a.to_faer().partial_piv_lu();
    let u = lu.U();
    if (0..a.rows).any(|i| u[(i, i)] == 0.0) {
        return None;
    }
    Some(lu)
}

/// The inverse, `numpy.linalg.inv`, or `None` where that raises.
pub fn inv(a: &Dense) -> Option<Dense> {
    if a.rows == 0 {
        return Some(Dense::zeros(0, 0));
    }
    let lu = lu(a)?;
    let mut x = Mat::<Number>::identity(a.rows, a.rows);
    lu.solve_in_place(x.as_mut());
    Some(Dense::from_faer(x.as_ref()))
}

/// `a⁻¹ b`, `numpy.linalg.solve`, or `None` where that raises.
pub fn solve(a: &Dense, b: &Dense) -> Option<Dense> {
    assert_eq!(a.rows, b.rows, "solve of mismatched shapes");
    if a.rows == 0 {
        return Some(Dense::zeros(0, b.cols));
    }
    let lu = lu(a)?;
    let mut x = b.to_faer();
    lu.solve_in_place(x.as_mut());
    Some(Dense::from_faer(x.as_ref()))
}

/// The eigenvalues of a symmetric matrix in ascending order,
/// `numpy.linalg.eigvalsh`. The lower triangle is read, as numpy reads it.
pub fn sym_eigenvalues(a: &Dense) -> Result<Vec<Number>, String> {
    if a.rows == 0 {
        return Ok(Vec::new());
    }
    a.to_faer()
        .self_adjoint_eigenvalues(Side::Lower)
        .map_err(|e| format!("the eigendecomposition did not converge: {e:?}"))
}

/// The eigenvalues in ascending order and the eigenvectors as columns of a
/// symmetric matrix, `numpy.linalg.eigh`, with each eigenvector's sign
/// pinned so its largest-magnitude component is positive, ties going to
/// the earliest component.
///
/// `v` and `-v` are equally valid eigenvectors, and which one a solver
/// returns is a convention of the build rather than a property of the
/// matrix, so without the pin the same model and data can report opposite
/// directions on two machines. The pin is scale-invariant. It fixes the
/// sign only: within a repeated eigenvalue's eigenspace the basis is still
/// arbitrary, and only the subspace the columns span reproduces.
pub fn sym_eigen(a: &Dense) -> Result<(Vec<Number>, Dense), String> {
    if a.rows == 0 {
        return Ok((Vec::new(), Dense::zeros(0, 0)));
    }
    let evd = a
        .to_faer()
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("the eigendecomposition did not converge: {e:?}"))?;
    let n = a.rows;
    let s = evd.S();
    let vals: Vec<Number> = (0..n).map(|i| s[i]).collect();
    let mut vecs = Dense::from_faer(evd.U());
    pin_eigenvector_signs(&mut vecs);
    Ok((vals, vecs))
}

/// Each column's largest-magnitude component made positive, ties going to
/// the earliest. An all-zero column is left alone. See [`sym_eigen`].
pub fn pin_eigenvector_signs(vecs: &mut Dense) {
    for j in 0..vecs.cols {
        let mut lead = 0;
        let mut best = Number::NEG_INFINITY;
        for i in 0..vecs.rows {
            let m = vecs.at(i, j).abs();
            if m > best {
                best = m;
                lead = i;
            }
        }
        if vecs.rows > 0 && vecs.at(lead, j) < 0.0 {
            for i in 0..vecs.rows {
                vecs.set(i, j, -vecs.at(i, j));
            }
        }
    }
}

/// The singular values in descending order.
fn singular_values(a: &Dense) -> Result<Vec<Number>, String> {
    if a.rows == 0 || a.cols == 0 {
        return Ok(Vec::new());
    }
    a.to_faer()
        .singular_values()
        .map_err(|e| format!("the singular value decomposition did not converge: {e:?}"))
}

/// The numerical rank, `numpy.linalg.matrix_rank`: the singular values
/// above `σ_max · max(rows, cols) · ε`.
pub fn rank(a: &Dense) -> Result<usize, String> {
    let sv = singular_values(a)?;
    let Some(&top) = sv.first() else {
        return Ok(0);
    };
    let tol = top * a.rows.max(a.cols) as Number * Number::EPSILON;
    Ok(sv.iter().filter(|&&s| s > tol).count())
}

/// An orthonormal basis of the null space of `a`'s rows, as columns.
///
/// With `a` the active constraint normals, the covariance lives in the span
/// of this basis and is zero along everything else. The rank is read off
/// the singular values at `max(rows, cols) · ε · σ_max`, so a constraint
/// listed twice costs one dimension rather than two. A matrix with no rows
/// has the whole space as its null space.
pub fn nullspace(a: &Dense) -> Result<Dense, String> {
    let n = a.cols;
    if a.rows == 0 {
        return Ok(Dense::identity(n));
    }
    let svd = a
        .to_faer()
        .svd()
        .map_err(|e| format!("the singular value decomposition did not converge: {e:?}"))?;
    let s = svd.S();
    let k = a.rows.min(n);
    let top = if k > 0 { s[0] } else { 1.0 };
    let tol = a.rows.max(n) as Number * Number::EPSILON * top;
    let rank = (0..k).filter(|&i| s[i] > tol).count();
    let v = svd.V();
    Ok(Dense::from_fn(n, n - rank, |i, j| v[(i, rank + j)]))
}

/// Whether a symmetric matrix is rank-deficient as a statement about
/// collinearity rather than units: the rank is taken of the diagonally
/// scaled form `A / (d dᵀ)` with `d = sqrt(|diag A|)`.
///
/// `matrix_rank`'s tolerance is relative to the largest singular value, so
/// it is not invariant under rescaling the coordinates against each other.
/// A covariance block carries the square of any unit spread between its
/// members, and two well-determined parameters about `1e8` apart in
/// magnitude push `cond(A)` past the tolerance on the spread alone. Scaling
/// to the correlation form first makes the test a scale-invariant ratio. A
/// zero diagonal entry leaves its row and column unscaled: there is no
/// scale to divide by, and a zero row is rank-deficient either way.
pub fn rank_deficient(a: &Dense) -> Result<bool, String> {
    let n = a.rows;
    if n == 0 {
        return Ok(false);
    }
    let d: Vec<Number> = a
        .diagonal()
        .iter()
        .map(|v| {
            let s = v.abs().sqrt();
            if s > 0.0 { s } else { 1.0 }
        })
        .collect();
    let scaled = Dense::from_fn(n, n, |i, j| a.at(i, j) / (d[i] * d[j]));
    Ok(rank(&scaled)? < n)
}

/// Whether a symmetric matrix has a genuinely negative eigenvalue, at a
/// tolerance relative to its own scale: the smallest eigenvalue below
/// `-1e-10 · max(1, max |a_ij|)`.
///
/// The scale is the largest entry, not each parameter's own, so unlike
/// [`rank_deficient`] this is not invariant to the parameters' units: a
/// parameter whose entries are small beside the block's largest can carry
/// a negative eigenvalue under the tolerance. This is the rule `_stats.py`
/// applied, ported unchanged; a negative variance on the diagonal is
/// reported separately whatever its size.
pub fn indefinite(a: &Dense) -> Result<bool, String> {
    if a.rows == 0 {
        return Ok(false);
    }
    let ev_min = sym_eigenvalues(a)?
        .into_iter()
        .fold(Number::INFINITY, Number::min);
    Ok(ev_min < -1e-10 * a.max_abs().max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Number, b: Number) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn the_inverse_inverts_and_a_zero_pivot_is_singular() {
        let a = Dense::from_fn(2, 2, |i, j| [[4.0, 1.0], [2.0, 3.0]][i][j]);
        let ai = inv(&a).expect("invertible");
        let id = a.mul(&ai);
        for i in 0..2 {
            for j in 0..2 {
                assert!(close(id.at(i, j), if i == j { 1.0 } else { 0.0 }));
            }
        }
        let s = Dense::from_fn(2, 2, |i, j| [[1.0, 2.0], [2.0, 4.0]][i][j]);
        assert!(inv(&s).is_none());
        assert!(solve(&s, &Dense::identity(2)).is_none());
    }

    #[test]
    fn eigenvectors_are_sign_pinned_and_values_ascend() {
        let a = Dense::from_fn(2, 2, |i, j| [[2.0, 1.0], [1.0, 2.0]][i][j]);
        let (w, v) = sym_eigen(&a).expect("eigen");
        assert!(close(w[0], 1.0) && close(w[1], 3.0));
        for j in 0..2 {
            let lead = if v.at(0, j).abs() >= v.at(1, j).abs() {
                0
            } else {
                1
            };
            assert!(v.at(lead, j) > 0.0, "column {j} is not pinned");
        }
        // the soft direction is (1, -1)/√2, its leading component positive
        assert!(close(v.at(0, 0), 0.5_f64.sqrt()) && close(v.at(1, 0), -(0.5_f64.sqrt())));
    }

    #[test]
    fn the_eigensolver_reads_the_lower_triangle() {
        // numpy's eigh and eigvalsh read the lower triangle and ignore the
        // upper one, so this is [[1, 0.5], [0.5, 1]] to both
        let a = Dense::from_fn(2, 2, |i, j| [[1.0, 99.0], [0.5, 1.0]][i][j]);
        let w = sym_eigenvalues(&a).expect("eigenvalues");
        let (we, _) = sym_eigen(&a).expect("eigen");
        for v in [w, we] {
            assert!(close(v[0], 0.5) && close(v[1], 1.5), "{v:?}");
        }
    }

    #[test]
    fn a_tie_pins_the_earliest_component_positive() {
        let mut v = Dense::from_fn(2, 1, |i, _| [-1.0, 1.0][i]);
        pin_eigenvector_signs(&mut v);
        assert_eq!(v.data, vec![1.0, -1.0]);
    }

    #[test]
    fn the_rank_uses_numpys_tolerance() {
        let a = Dense::from_fn(3, 3, |i, j| {
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]][i][j]
        });
        assert_eq!(rank(&a).unwrap(), 2);
        assert_eq!(rank(&Dense::identity(3)).unwrap(), 3);
        assert_eq!(rank(&Dense::zeros(2, 2)).unwrap(), 0);
        // numpy's tolerance carries the dimension: at max(2, 2)·ε = 4.4e-16,
        // a singular value of 3e-16 is below it
        let tiny = Dense::from_fn(2, 2, |i, j| if i == j { [1.0, 3e-16][i] } else { 0.0 });
        assert_eq!(rank(&tiny).unwrap(), 1);
    }

    #[test]
    fn rank_deficiency_reads_collinearity_not_units() {
        // well determined, members 1e8 apart in scale
        let a = Dense::from_fn(2, 2, |i, j| [[1e16, 1e7], [1e7, 1.0]][i][j]);
        assert!(!rank_deficient(&a).unwrap());
        let d = Dense::from_fn(2, 2, |i, j| [[1.0, 1.0], [1.0, 1.0]][i][j]);
        assert!(rank_deficient(&d).unwrap());
    }

    #[test]
    fn the_null_space_is_orthonormal_and_annihilated() {
        let a = Dense::from_fn(1, 3, |_, j| [1.0, 1.0, 0.0][j]);
        let z = nullspace(&a).unwrap();
        assert_eq!((z.rows, z.cols), (3, 2));
        let az = a.mul(&z);
        assert!(az.max_abs() < 1e-14);
        let ztz = z.t().mul(&z);
        for i in 0..2 {
            for j in 0..2 {
                assert!(close(ztz.at(i, j), if i == j { 1.0 } else { 0.0 }));
            }
        }
        // a row listed twice costs one dimension
        let twice = Dense::from_fn(2, 3, |_, j| [1.0, 1.0, 0.0][j]);
        assert_eq!(nullspace(&twice).unwrap().cols, 2);
        assert_eq!(nullspace(&Dense::zeros(0, 3)).unwrap(), Dense::identity(3));
    }

    #[test]
    fn indefiniteness_is_relative_to_scale() {
        let pd = Dense::from_fn(2, 2, |i, j| [[2.0, 0.0], [0.0, 1e-12]][i][j]);
        assert!(!indefinite(&pd).unwrap());
        let neg = Dense::from_fn(2, 2, |i, j| [[2.0, 0.0], [0.0, -1e-3]][i][j]);
        assert!(indefinite(&neg).unwrap());
        // a negative eigenvalue at roundoff of the matrix's scale is not one
        let round = Dense::from_fn(2, 2, |i, j| [[2.0, 0.0], [0.0, -1e-14]][i][j]);
        assert!(!indefinite(&round).unwrap());
    }
}
