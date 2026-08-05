//! PyO3 binding for FERAL's unsymmetric sparse LU (general `A x = b`).
//!
//! POUNCE's NLP solver factorizes *symmetric* KKT saddle systems. A square
//! root-find (e.g. a BVP collocation system) is better served by a direct
//! Newton iteration on the unsymmetric `N x N` Jacobian — exactly what
//! SciPy's `solve_bvp` does. FERAL (POUNCE's pure-Rust linear backend)
//! ships an unsymmetric sparse LU; this class exposes it so the Python
//! side can run that Newton loop without going through the interior-point
//! method (and its `2N` saddle system).
//!
//! Usage: build once over a fixed sparsity pattern (`rows`, `cols` in
//! 0-based COO), then `factor(values)` / `solve(b)` repeatedly as the
//! Newton iterates change the values but not the structure. `solve` does
//! `A x = b` (FERAL's `ftran`); `solve_transpose` does `Aᵀ x = b`
//! (`btran`), which the implicit-differentiation backward needs.

use feral::{LuParams, SparseColMatrix, SparseLu, SparseLuSymbolic};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Sendable, for the reason [`crate::dense_lu::PyDenseLu`] gives: FERAL's
/// symbolic and numeric factors are owned data with no thread affinity,
/// and `unsendable` bought only pyo3's uncatchable thread check
/// (pounce#477).
#[pyclass(name = "SparseLU", module = "pounce._pounce")]
pub struct PySparseLu {
    n: usize,
    nnz: usize,
    /// Per-column list of `(row, index-into-values)` so a fresh value
    /// vector is scattered into FERAL's column-sparse format in O(nnz).
    col_entries: Vec<Vec<(usize, usize)>>,
    symbolic: Option<SparseLuSymbolic>,
    lu: Option<SparseLu>,
}

#[pymethods]
impl PySparseLu {
    /// Build a reusable LU over the fixed sparsity pattern of an `n x n`
    /// matrix. `rows` / `cols` are 0-based COO coordinates; the matching
    /// values are supplied later by :meth:`factor`.
    #[new]
    fn new(n: usize, rows: Vec<i64>, cols: Vec<i64>) -> PyResult<Self> {
        if rows.len() != cols.len() {
            return Err(PyValueError::new_err(
                "SparseLU: rows and cols must have equal length",
            ));
        }
        let mut col_entries: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (t, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
            if r < 0 || c < 0 || (r as usize) >= n || (c as usize) >= n {
                return Err(PyValueError::new_err(format!(
                    "SparseLU: entry ({r}, {c}) out of range for n={n}"
                )));
            }
            col_entries[c as usize].push((r as usize, t));
        }
        Ok(Self {
            n,
            nnz: rows.len(),
            col_entries,
            symbolic: None,
            lu: None,
        })
    }

    /// Numerically (re)factor `A` from `values` aligned to the
    /// `(rows, cols)` pattern. The symbolic analysis is computed once and
    /// reused across calls (the pattern is fixed), so repeated Newton
    /// factorizations pay only the numeric cost. Raises ``RuntimeError``
    /// if the matrix is singular.
    fn factor(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()> {
        let v = values.as_slice()?;
        if v.len() != self.nnz {
            return Err(PyValueError::new_err(format!(
                "SparseLU.factor: values has length {} but the pattern has {} nonzeros",
                v.len(),
                self.nnz
            )));
        }
        let mut cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(self.n);
        for c in 0..self.n {
            let entries = &self.col_entries[c];
            let mut col = Vec::with_capacity(entries.len());
            for &(r, t) in entries {
                col.push((r, v[t]));
            }
            cols.push(col);
        }
        let m = SparseColMatrix::from_sparse_columns(self.n, &cols)
            .map_err(|e| PyValueError::new_err(format!("SparseLU: bad matrix: {e:?}")))?;
        if self.symbolic.is_none() {
            let s = SparseLuSymbolic::analyze(&m)
                .map_err(|e| PyValueError::new_err(format!("SparseLU: analyze failed: {e:?}")))?;
            self.symbolic = Some(s);
        }
        let symbolic = self.symbolic.as_ref().unwrap();
        let lu = SparseLu::factor(&m, symbolic, LuParams::default()).map_err(|e| {
            PyRuntimeError::new_err(format!("SparseLU: factorization failed (singular?): {e:?}"))
        })?;
        self.lu = Some(lu);
        Ok(())
    }

    /// Solve `A x = b`, returning `x`. Requires a prior :meth:`factor`.
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let lu = self
            .lu
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("SparseLU.solve: call factor() first"))?;
        let mut rhs = b.as_slice()?.to_vec();
        if rhs.len() != lu.dim() {
            return Err(PyValueError::new_err(format!(
                "SparseLU.solve: rhs length {} != n {}",
                rhs.len(),
                lu.dim()
            )));
        }
        lu.ftran(&mut rhs)
            .map_err(|e| PyRuntimeError::new_err(format!("SparseLU.solve: ftran failed: {e:?}")))?;
        Ok(rhs.into_pyarray_bound(py))
    }

    /// Solve `Aᵀ x = b`, returning `x`. Requires a prior :meth:`factor`.
    fn solve_transpose<'py>(
        &mut self,
        py: Python<'py>,
        b: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let lu = self.lu.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("SparseLU.solve_transpose: call factor() first")
        })?;
        let mut rhs = b.as_slice()?.to_vec();
        if rhs.len() != lu.dim() {
            return Err(PyValueError::new_err(format!(
                "SparseLU.solve_transpose: rhs length {} != n {}",
                rhs.len(),
                lu.dim()
            )));
        }
        lu.btran(&mut rhs).map_err(|e| {
            PyRuntimeError::new_err(format!("SparseLU.solve_transpose: btran failed: {e:?}"))
        })?;
        Ok(rhs.into_pyarray_bound(py))
    }

    /// Solve `A X = B` for `n_rhs` right-hand sides against the single held
    /// factor. ``b_flat`` is the row-major flatten of the ``(n_rhs, n)``
    /// RHS block; the result is the matching flat ``(n_rhs, n)`` solution.
    /// One factorization, many back-solves — the multi-RHS path the
    /// implicit-diff Jacobian (``jax.jacobian`` / batched VJP) needs.
    fn solve_many<'py>(
        &mut self,
        py: Python<'py>,
        b_flat: PyReadonlyArray1<f64>,
        n_rhs: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.multi(py, b_flat, n_rhs, false)
    }

    /// Like :meth:`solve_many` but solves `Aᵀ X = B` (the VJP direction).
    fn solve_transpose_many<'py>(
        &mut self,
        py: Python<'py>,
        b_flat: PyReadonlyArray1<f64>,
        n_rhs: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.multi(py, b_flat, n_rhs, true)
    }
}

impl PySparseLu {
    fn multi<'py>(
        &mut self,
        py: Python<'py>,
        b_flat: PyReadonlyArray1<f64>,
        n_rhs: usize,
        transpose: bool,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let lu = self
            .lu
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("SparseLU: call factor() first"))?;
        let nn = lu.dim();
        let b = b_flat.as_slice()?;
        if b.len() != n_rhs * nn {
            return Err(PyValueError::new_err(format!(
                "SparseLU multi-solve: rhs length {} != n_rhs ({}) * n ({}) = {}",
                b.len(),
                n_rhs,
                nn,
                n_rhs * nn
            )));
        }
        let mut out = vec![0.0_f64; b.len()];
        for r in 0..n_rhs {
            let mut rhs = b[r * nn..(r + 1) * nn].to_vec();
            let res = if transpose {
                lu.btran(&mut rhs)
            } else {
                lu.ftran(&mut rhs)
            };
            res.map_err(|e| {
                PyRuntimeError::new_err(format!("SparseLU multi-solve failed: {e:?}"))
            })?;
            out[r * nn..(r + 1) * nn].copy_from_slice(&rhs);
        }
        Ok(out.into_pyarray_bound(py))
    }
}

#[cfg(test)]
mod tests {
    /// The factor must stay movable across threads (pounce#477); see
    /// [`super::PySparseLu`] for why `unsendable` is the worse default.
    #[test]
    fn py_sparse_lu_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::PySparseLu>();
    }
}
