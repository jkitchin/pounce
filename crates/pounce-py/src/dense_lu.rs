//! PyO3 binding for faer's dense partial-pivoting LU (general `A x = b` on a
//! small **dense** matrix).
//!
//! `pounce.ode`'s Radau IIA(5) integrator factorizes the `(3n×3n)` stage
//! operator `I⊗M − h(A⊗J)` and the `(n×n)` error operator every step. These
//! are dense and tiny, but the engine previously ran them through FERAL's
//! *sparse* LU (over a full dense pattern), which (a) pays sparse symbolic /
//! supernodal overhead on a 6×6/9×9 block and (b) is stricter than LAPACK —
//! it hard-fails as `SingularBasis` on the ill-conditioned (large-`h`) stage
//! matrices a stiff/DAE problem reaches on its slow manifold, crashing the
//! integration (pounce#175).
//!
//! A dense partial-pivoting LU is the right tool: faster on these blocks, no
//! symbolic phase, and — like LAPACK / SciPy's `Radau` — it always completes
//! (a genuinely singular matrix shows up as `inf`/`nan` in the *solve*, which
//! the Newton/error control already handles by shrinking the step) rather than
//! refusing to factor.
//!
//! Interface mirrors [`crate::sparse_lu::PySparseLu`] (`factor(values)` /
//! `solve(b)`) so the Python side swaps backends with no call-site changes.
//! Values are row-major (C-order) `n*n`, matching `numpy`'s default flatten.

use faer::linalg::solvers::{PartialPivLu, Solve};
use faer::Mat;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "DenseLU", module = "pounce._pounce", unsendable)]
pub struct PyDenseLu {
    n: usize,
    lu: Option<PartialPivLu<f64>>,
}

#[pymethods]
impl PyDenseLu {
    /// Build a reusable dense LU for an `n × n` matrix. Numerical values are
    /// supplied later by [`Self::factor`].
    #[new]
    fn new(n: usize) -> Self {
        Self { n, lu: None }
    }

    /// Factor `A` from its **row-major** `n*n` `values`. Dense
    /// partial-pivoting LU has no symbolic phase and never rejects a matrix as
    /// "singular": a rank-deficient `A` only surfaces as `inf`/`nan` in a
    /// later :meth:`solve`.
    fn factor(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()> {
        let v = values.as_slice()?;
        let n = self.n;
        if v.len() != n * n {
            return Err(PyValueError::new_err(format!(
                "DenseLU.factor: values has length {} but expected n*n = {}",
                v.len(),
                n * n
            )));
        }
        // Row-major (C-order) input → faer's column-major `Mat`:
        // logical `A[(i, j)] = v[i*n + j]`.
        let a = Mat::from_fn(n, n, |i, j| v[i * n + j]);
        self.lu = Some(PartialPivLu::new(a.as_ref()));
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
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("DenseLU.solve: call factor() first"))?;
        let bs = b.as_slice()?;
        if bs.len() != self.n {
            return Err(PyValueError::new_err(format!(
                "DenseLU.solve: rhs length {} != n {}",
                bs.len(),
                self.n
            )));
        }
        let mut rhs = Mat::from_fn(self.n, 1, |i, _| bs[i]);
        lu.solve_in_place(rhs.as_mut());
        let out: Vec<f64> = (0..self.n).map(|i| rhs[(i, 0)]).collect();
        Ok(out.into_pyarray_bound(py))
    }
}
