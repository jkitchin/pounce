//! `.nl` loader for Python.
//!
//! [`read_nl`] parses an AMPL `.nl` file through pounce's own reader
//! ([`pounce_nl::nl_reader::read_nl_file`]) and returns a [`PyNlProblem`]
//! that hands back the model's evaluators: objective, gradient, and
//! Lagrangian Hessian plus the constraint values and Jacobian. The heavy
//! lifting (the reverse-mode AD tape, sparsity, external functions) is the
//! same [`pounce_nl::nl_reader::NlTnlp`] the CLI solves with, so a Python
//! caller sees exactly the derivatives pounce itself uses.
//!
//! ```python
//! import pounce, numpy as np
//! p = pounce.read_nl("model.nl")
//! x = np.asarray(p.x0)
//! f  = p.objective(x)              # float
//! g  = p.gradient(x)               # ndarray[n]
//! c  = p.constraints(x)            # ndarray[m]
//! Jr, Jc = p.jacobian_structure()  # COO rows / cols (0-based)
//! Jv = p.jacobian(x)               # ndarray[nnz_jac], aligned to (Jr, Jc)
//! Hr, Hc = p.hessian_structure()   # lower-triangle rows / cols
//! Hv = p.hessian(x)                # ndarray[nnz_h] of the Lagrangian Hessian
//! ```
//!
//! Values follow the solver's (minimization) convention: for a `.nl` whose
//! original sense is `maximize`, the objective/gradient/Hessian are negated
//! so that minimizing them solves the model. The original sense is exposed
//! as [`PyNlProblem::minimize`].

use std::cell::RefCell;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pounce_common::types::{Index, Number};
use pounce_nl::nl_reader::{
    Expr, NlProblem, NlTnlp, NlVariation, parse_nl_text as parse_nl_text_rs, read_nl_file,
};
use pounce_nlp::tnlp::{SparsityRequest, TNLP};

use crate::nl_expr::{INLINE_DEPTH, MAX_DEPTH, expr_depth, on_deep_stack, on_worker_stack};

/// A `.nl` model loaded through pounce's reader, exposing its evaluators.
//
// **Sendable.** `NlTnlp` is `Send` (its CSE nodes went `Arc` for
// pounce#126's batched solving) and nothing else here is thread-affine,
// so this pyclass carries no `unsendable` marker: one `NlProblem` can be
// built on one thread and evaluated — or dropped — on another, which is
// what a threaded host (a branch-and-bound worker pool) needs of a shared
// evaluator.
//
// It stayed `unsendable` for a while as a conservative default, and that
// default was actively harmful (pounce#477): pyo3's thread check raises
// `PanicException`, which derives from `BaseException` and so slips past
// every `except Exception` in the host — a cross-thread call surfaced not
// as an error but as a wrong answer, and the *drop* path tripped it even
// for code that never used an instance cross-thread (whichever thread
// runs the GC inherits the object).
//
// Concurrency is still the GIL's: every `&self` method below borrows
// `tnlp` and evaluates without releasing the GIL, so the `RefCell`
// borrows of two threads can never overlap. Any future method that wants
// to release the GIL around an evaluation must first take the `NlTnlp`
// out of the `RefCell` (as `clone_tnlp` does for the batch path) rather
// than hold a borrow across the release.
#[pyclass(module = "pounce", name = "NlProblem")]
pub struct PyNlProblem {
    tnlp: RefCell<NlTnlp>,
    n: usize,
    m: usize,
    nnz_jac: usize,
    nnz_h: usize,
    /// Nesting depth of the expressions inside `tnlp`. `NlTnlp` keeps the
    /// `Expr` trees it taped, so copying or dropping this pyclass recurses
    /// once per level — the same hazard `NlExpr` guards, reached through
    /// the problem instead (pounce#472).
    expr_depth: u32,
    // Metadata captured before `prob` was moved into `NlTnlp`.
    minimize: bool,
    obj_constant: Number,
    x0: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    var_names: Vec<String>,
    con_names: Vec<String>,
}

/// Decode a 1-D float input (NumPy `float64` array or any float sequence)
/// into a `Vec<f64>` of the expected length.
fn decode_vec(val: &Bound<'_, PyAny>, expected: usize, what: &str) -> PyResult<Vec<Number>> {
    if let Ok(arr) = val.downcast::<PyArray1<Number>>() {
        let len = arr.len();
        if len != expected {
            return Err(PyValueError::new_err(format!(
                "{what}: expected length {expected}, got {len}"
            )));
        }
        // `as_slice` requires C-contiguity, but a strided view (`x[::2]`,
        // a column of a 2-D array) is still a `PyArray1<f64>` and reaches
        // this branch. Falling back to the strided view keeps such an `x`
        // working, and — more to the point — keeps the failure mode of a
        // bad input a named `{what}` error rather than numpy's bare
        // "The given array is not contiguous", which says nothing about
        // which argument was at fault.
        return Ok(match unsafe { arr.as_slice() } {
            Ok(s) => s.to_vec(),
            Err(_) => arr.readonly().as_array().iter().copied().collect(),
        });
    }
    let mut out = Vec::with_capacity(expected);
    for item in val.iter()? {
        out.push(item?.extract::<Number>()?);
    }
    if out.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{what}: expected length {expected}, got {}",
            out.len()
        )));
    }
    Ok(out)
}

/// One or more Hessian-vector-product directions, decoded into the
/// column-major `n × k` layout [`NlTnlp::hessian_vector_products`] wants.
struct Directions {
    /// `n * k` values; direction `c` occupies `data[c*n .. (c+1)*n]`.
    data: Vec<Number>,
    k: usize,
    /// `true` when the caller passed a single 1-D vector, so the result
    /// should come back 1-D rather than as an `(n, 1)` column.
    single: bool,
}

/// Decode `v` into [`Directions`], accepting a dense vector, a dense
/// `(n, k)` array, or any SciPy sparse vector / matrix.
///
/// SciPy sparse is recognized by duck-typing `.toarray()` rather than by
/// importing scipy: this crate must build and run without it, and any
/// object that can hand back a dense array is a legitimate input anyway.
/// Densifying is the honest move — `v` is only `n × k`, and the sparsity
/// worth exploiting (the model's) lives in the tapes, not here.
///
/// The shape rule is deliberately strict: `(n,)` or `(n, k)`. A SciPy row
/// vector densifies to `(1, n)`, which is *not* accepted, because for a
/// square-ish block it would be indistinguishable from `k` directions of
/// the wrong length. The error says what to pass instead.
fn decode_directions(v: &Bound<'_, PyAny>, n: usize, what: &str) -> PyResult<Directions> {
    let py = v.py();

    let dense = if v.hasattr("toarray")? {
        v.call_method0("toarray")?
    } else {
        v.clone()
    };

    // Route everything through `numpy.asarray(..., float64)` so lists,
    // tuples, integer arrays, and strided views (a column `V[:, 0]`, a
    // slice `w[::2]`) all arrive as a plain float64 array. numpy is a hard
    // dependency of the package and already linked by this extension.
    let np = py.import_bound("numpy")?;
    let arr = np
        .getattr("asarray")?
        .call1((&dense, np.getattr("float64")?))
        .map_err(|e| {
            PyValueError::new_err(format!("{what}: could not read as a float array ({e})"))
        })?;

    let ndim: usize = arr.getattr("ndim")?.extract()?;
    match ndim {
        1 => {
            let a = arr
                .downcast::<PyArray1<Number>>()
                .map_err(|_| PyValueError::new_err(format!("{what}: expected a float64 array")))?;
            let got = a.len();
            if got != n {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected length {n}, got {got}"
                )));
            }
            // `as_slice` needs C-contiguity; a strided view is still a
            // valid `PyArray1<f64>`, so copy through the strided view
            // rather than erroring on it.
            let data = match unsafe { a.as_slice() } {
                Ok(s) => s.to_vec(),
                Err(_) => a.readonly().as_array().iter().copied().collect(),
            };
            Ok(Directions {
                data,
                k: 1,
                single: true,
            })
        }
        2 => {
            let a = arr
                .downcast::<PyArray2<Number>>()
                .map_err(|_| PyValueError::new_err(format!("{what}: expected a float64 array")))?;
            let ro = a.readonly();
            let view = ro.as_array();
            let (rows, k) = (view.shape()[0], view.shape()[1]);
            if rows != n {
                return Err(PyValueError::new_err(format!(
                    "{what}: expected shape ({n},) or ({n}, k), got ({rows}, {k}); \
                     a row-vector direction needs transposing"
                )));
            }
            let mut data = Vec::with_capacity(n * k);
            for c in 0..k {
                for i in 0..n {
                    data.push(view[[i, c]]);
                }
            }
            Ok(Directions {
                data,
                k,
                single: false,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "{what}: expected a 1-D or 2-D array, got {other} dimensions"
        ))),
    }
}

#[pymethods]
impl PyNlProblem {
    /// Number of variables.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    /// Number of constraints.
    #[getter]
    fn m(&self) -> usize {
        self.m
    }

    /// Number of structurally non-zero Jacobian entries.
    #[getter]
    fn nnz_jac(&self) -> usize {
        self.nnz_jac
    }

    /// Number of stored (lower-triangle) Lagrangian-Hessian entries.
    #[getter]
    fn nnz_hess(&self) -> usize {
        self.nnz_h
    }

    /// `True` if the model's original sense is minimize, `False` if it was
    /// `maximize` (in which case the returned objective is negated).
    #[getter]
    fn minimize(&self) -> bool {
        self.minimize
    }

    /// Constant offset of the objective.
    #[getter]
    fn obj_constant(&self) -> Number {
        self.obj_constant
    }

    /// Starting point from the `.nl` file (length `n`).
    #[getter]
    fn x0<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.x0.clone().into_pyarray_bound(py)
    }

    /// Variable lower bounds (length `n`).
    #[getter]
    fn x_l<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.x_l.clone().into_pyarray_bound(py)
    }

    /// Variable upper bounds (length `n`).
    #[getter]
    fn x_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.x_u.clone().into_pyarray_bound(py)
    }

    /// Constraint lower bounds (length `m`).
    #[getter]
    fn g_l<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.g_l.clone().into_pyarray_bound(py)
    }

    /// Constraint upper bounds (length `m`).
    #[getter]
    fn g_u<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Number>> {
        self.g_u.clone().into_pyarray_bound(py)
    }

    /// Variable names from the sibling `.col` file (empty if absent).
    #[getter]
    fn var_names(&self) -> Vec<String> {
        self.var_names.clone()
    }

    /// Constraint names from the sibling `.row` file (empty if absent).
    #[getter]
    fn con_names(&self) -> Vec<String> {
        self.con_names.clone()
    }

    /// Objective value `f(x)`.
    fn objective(&self, x: &Bound<'_, PyAny>) -> PyResult<Number> {
        let xv = decode_vec(x, self.n, "objective: x")?;
        self.tnlp
            .borrow_mut()
            .eval_f(&xv, true)
            .ok_or_else(|| PyValueError::new_err("objective evaluation failed"))
    }

    /// Objective gradient `∇f(x)` (length `n`).
    fn gradient<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "gradient: x")?;
        let mut grad = vec![0.0; self.n];
        if !self.tnlp.borrow_mut().eval_grad_f(&xv, true, &mut grad) {
            return Err(PyValueError::new_err("gradient evaluation failed"));
        }
        Ok(grad.into_pyarray_bound(py))
    }

    /// Constraint values `g(x)` (length `m`).
    fn constraints<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "constraints: x")?;
        let mut g = vec![0.0; self.m];
        if !self.tnlp.borrow_mut().eval_g(&xv, true, &mut g) {
            return Err(PyValueError::new_err("constraint evaluation failed"));
        }
        Ok(g.into_pyarray_bound(py))
    }

    /// Jacobian sparsity as 0-based COO `(rows, cols)`, each length
    /// `nnz_jac`. Aligns entry-for-entry with [`Self::jacobian`].
    fn jacobian_structure<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<Index>>, Bound<'py, PyArray1<Index>>)> {
        let mut rows = vec![0 as Index; self.nnz_jac];
        let mut cols = vec![0 as Index; self.nnz_jac];
        let ok = self.tnlp.borrow_mut().eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut rows,
                jcol: &mut cols,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("jacobian structure failed"));
        }
        Ok((rows.into_pyarray_bound(py), cols.into_pyarray_bound(py)))
    }

    /// Jacobian values at `x` (length `nnz_jac`), aligned to
    /// [`Self::jacobian_structure`].
    fn jacobian<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "jacobian: x")?;
        let mut values = vec![0.0; self.nnz_jac];
        let ok = self.tnlp.borrow_mut().eval_jac_g(
            Some(&xv),
            true,
            SparsityRequest::Values {
                values: &mut values,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("jacobian evaluation failed"));
        }
        Ok(values.into_pyarray_bound(py))
    }

    /// Lower-triangle Lagrangian-Hessian sparsity as 0-based COO
    /// `(rows, cols)`, each length `nnz_hess`. Aligns with
    /// [`Self::hessian`].
    fn hessian_structure<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<Index>>, Bound<'py, PyArray1<Index>>)> {
        let mut rows = vec![0 as Index; self.nnz_h];
        let mut cols = vec![0 as Index; self.nnz_h];
        let ok = self.tnlp.borrow_mut().eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut rows,
                jcol: &mut cols,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("hessian structure failed"));
        }
        Ok((rows.into_pyarray_bound(py), cols.into_pyarray_bound(py)))
    }

    /// Lower-triangle of the Lagrangian Hessian
    /// `obj_factor·∇²f + Σ_i lam_i·∇²g_i` at `x` (length `nnz_hess`),
    /// aligned to [`Self::hessian_structure`].
    ///
    /// `lam` defaults to zeros (the objective Hessian alone); `obj_factor`
    /// defaults to 1.0.
    #[pyo3(signature = (x, lam=None, obj_factor=1.0))]
    fn hessian<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        lam: Option<&Bound<'_, PyAny>>,
        obj_factor: Number,
    ) -> PyResult<Bound<'py, PyArray1<Number>>> {
        let xv = decode_vec(x, self.n, "hessian: x")?;
        let lamv = match lam {
            Some(l) => decode_vec(l, self.m, "hessian: lam")?,
            None => vec![0.0; self.m],
        };
        let mut values = vec![0.0; self.nnz_h];
        let ok = self.tnlp.borrow_mut().eval_h(
            Some(&xv),
            true,
            obj_factor,
            Some(&lamv),
            true,
            SparsityRequest::Values {
                values: &mut values,
            },
        );
        if !ok {
            return Err(PyValueError::new_err("hessian evaluation failed"));
        }
        Ok(values.into_pyarray_bound(py))
    }

    /// Hessian-vector product of the Lagrangian:
    /// `(obj_factor·∇²f + Σ_i lam_i·∇²g_i) · v`.
    ///
    /// Matrix-free: one forward-over-reverse AD pass per tape, seeded with
    /// `v` directly. [`Self::hessian`] instead runs one such pass *per
    /// Hessian color* and decodes the compressed columns into the sparse
    /// lower triangle, so on a model where materializing `∇²L` is
    /// impractical — the case a Newton–Krylov / truncated-CG step is built
    /// for — this is the cheaper call by the chromatic number of the
    /// coloring.
    ///
    /// `v` may be:
    ///
    /// * a dense length-`n` vector — NumPy array (any dtype or stride),
    ///   list, or other sequence — giving a length-`n` result;
    /// * a dense `(n, k)` array of `k` directions, giving an `(n, k)`
    ///   result;
    /// * a SciPy sparse `(n,)` vector, `(n, 1)` column, or `(n, k)` matrix.
    ///
    /// The shape rule is the same for dense and sparse: `(n,)` or
    /// `(n, k)`. A `(1, n)` *row* raises — for a square-ish block it is
    /// indistinguishable from `k` directions of the wrong length, so it is
    /// reported rather than guessed at. Note this catches
    /// `csr_matrix(v_1d)`, which SciPy shapes as `(1, n)`; pass
    /// `v_1d[:, None]`, or use the 1-D sparse *array* API
    /// (`coo_array(v_1d)`, SciPy >= 1.14), which is genuinely `(n,)`.
    ///
    /// A sparse `v` is densified on the way in and an all-zero direction is
    /// skipped, so a mostly-empty block costs only the columns that carry
    /// signal. The sparsity that actually pays is the *model's*, and the
    /// tapes exploit that regardless of how `v` arrives: each pass is
    /// O(tape ops), never O(n²).
    ///
    /// The result is always dense. `∇²L · v` is dense in general even when
    /// both `∇²L` and `v` are sparse — a sparse return type would promise
    /// an economy this product does not have. Callers who want the sparse
    /// Hessian itself already have [`Self::hessian`] +
    /// [`Self::hessian_structure`].
    ///
    /// `lam` defaults to zeros (the objective Hessian alone); `obj_factor`
    /// defaults to 1.0. Sign convention matches [`Self::hessian`].
    #[pyo3(signature = (x, v, lam=None, obj_factor=1.0))]
    fn hessian_vector_product<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
        v: &Bound<'_, PyAny>,
        lam: Option<&Bound<'_, PyAny>>,
        obj_factor: Number,
    ) -> PyResult<Bound<'py, PyAny>> {
        let xv = decode_vec(x, self.n, "hessian_vector_product: x")?;
        let dirs = decode_directions(v, self.n, "hessian_vector_product: v")?;
        let lamv = match lam {
            Some(l) => Some(decode_vec(l, self.m, "hessian_vector_product: lam")?),
            None => None,
        };
        let mut out = vec![0.0; self.n * dirs.k];
        self.tnlp
            .borrow_mut()
            .hessian_vector_products(
                &xv,
                &dirs.data,
                dirs.k,
                obj_factor,
                lamv.as_deref(),
                &mut out,
            )
            .map_err(PyValueError::new_err)?;

        if dirs.single {
            // A 1-D `v` in, a 1-D result out — the Krylov-callback shape.
            Ok(out.into_pyarray_bound(py).into_any())
        } else {
            // `out` is column-major (direction `c` at `c*n`); `Array2`
            // wants row-major, so index rather than reshape.
            let arr = Array2::from_shape_fn((self.n, dirs.k), |(i, c)| out[c * self.n + i]);
            Ok(arr.into_pyarray_bound(py).into_any())
        }
    }

    /// Clone this model with per-instance overrides applied — the
    /// "one structure, many bound / starting-point variations" case of
    /// batched solving (pounce#126): parametric sweeps, multi-start,
    /// or branch-and-bound nodes that only tighten variable bounds.
    /// The parsed expression DAG / AD tapes are shared structure and
    /// cheap to clone; only the named vectors are replaced. Arguments
    /// left as `None` keep this model's values.
    #[pyo3(signature = (x0=None, x_l=None, x_u=None, g_l=None, g_u=None))]
    fn variant(
        &self,
        x0: Option<&Bound<'_, PyAny>>,
        x_l: Option<&Bound<'_, PyAny>>,
        x_u: Option<&Bound<'_, PyAny>>,
        g_l: Option<&Bound<'_, PyAny>>,
        g_u: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNlProblem> {
        let dec = |v: Option<&Bound<'_, PyAny>>, len: usize, what: &str| {
            v.map(|b| decode_vec(b, len, what)).transpose()
        };
        let variation = NlVariation {
            x0: dec(x0, self.n, "variant: x0")?,
            x_l: dec(x_l, self.n, "variant: x_l")?,
            x_u: dec(x_u, self.n, "variant: x_u")?,
            g_l: dec(g_l, self.m, "variant: g_l")?,
            g_u: dec(g_u, self.m, "variant: g_u")?,
        };
        // `variant` copies the model, expressions included, so it carries
        // the same recursion depth (pounce#472).
        let tnlp = {
            let held = self.tnlp.borrow();
            let src: &NlTnlp = &held;
            on_deep_stack(self.expr_depth, move || src.variant(&variation))
                .map_err(PyValueError::new_err)?
        };
        PyNlProblem::from_tnlp(tnlp, "variant", self.expr_depth)
    }

    fn __repr__(&self) -> String {
        format!(
            "NlProblem(n={}, m={}, nnz_jac={}, nnz_hess={}, minimize={})",
            self.n, self.m, self.nnz_jac, self.nnz_h, self.minimize
        )
    }
}

impl PyNlProblem {
    /// Build the pyclass around an owned `NlTnlp`, capturing the
    /// metadata the getters serve. `what` labels error messages.
    pub(crate) fn from_tnlp(
        mut tnlp: NlTnlp,
        what: &str,
        expr_depth: u32,
    ) -> PyResult<PyNlProblem> {
        let info = tnlp
            .get_nlp_info()
            .ok_or_else(|| PyValueError::new_err(format!("{what}: get_nlp_info returned None")))?;
        let prob = tnlp.problem();
        let (n, m) = (prob.n, prob.m);
        let minimize = prob.minimize;
        let obj_constant = prob.obj_constant;
        let x0 = prob.x0.clone();
        let x_l = prob.x_l.clone();
        let x_u = prob.x_u.clone();
        let g_l = prob.g_l.clone();
        let g_u = prob.g_u.clone();
        let var_names = prob.var_names.clone();
        let con_names = prob.con_names.clone();
        Ok(PyNlProblem {
            tnlp: RefCell::new(tnlp),
            n,
            m,
            nnz_jac: info.nnz_jac_g as usize,
            nnz_h: info.nnz_h_lag as usize,
            expr_depth,
            minimize,
            obj_constant,
            x0,
            x_l,
            x_u,
            g_l,
            g_u,
            var_names,
            con_names,
        })
    }

    /// Owned copy of the evaluator for the batch path: the clone (not
    /// the pyclass) moves to a rayon worker. Cheap relative to a
    /// solve — tapes are flat `Vec`s of ops.
    pub(crate) fn clone_tnlp(&self) -> NlTnlp {
        let held = self.tnlp.borrow();
        let tnlp: &NlTnlp = &held;
        // The clone copies the `Expr` trees along with the tapes, so it
        // recurses once per level of nesting (pounce#472).
        on_deep_stack(self.expr_depth, move || tnlp.clone())
    }

    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    /// Per-constraint equality mask (`g_l[i] == g_u[i]`), used by the batch
    /// info-dict builder to reproduce the single-solve `active_constraints`
    /// classification (equalities are always active).
    pub(crate) fn equality_mask(&self) -> Vec<bool> {
        self.g_l
            .iter()
            .zip(&self.g_u)
            .map(|(l, u)| l == u)
            .collect()
    }
}

/// Tear the expression trees down on a stack sized for them, for a
/// problem deep enough that the recursive drop could overrun the
/// collecting thread's (pounce#472). The tapes the evaluators actually
/// use are flat `Vec`s and drop normally.
impl Drop for PyNlProblem {
    fn drop(&mut self) {
        if self.expr_depth <= INLINE_DEPTH {
            return;
        }
        let prob = self.tnlp.get_mut().problem_mut();
        let mut doomed = Vec::with_capacity(prob.con_nonlinear.len() + 1);
        doomed.push(std::mem::replace(&mut prob.obj_nonlinear, Expr::Const(0.0)));
        for c in &mut prob.con_nonlinear {
            doomed.push(std::mem::replace(c, Expr::Const(0.0)));
        }
        on_deep_stack(self.expr_depth, move || drop(doomed));
    }
}

/// Depth of the deepest expression in `prob`, and a `ValueError`-shaped
/// message when that is past what the machinery will carry.
///
/// A parsed model arrives already built, so the construction-time cap
/// `NlExpr` enforces cannot cover it — this is the same limit applied at
/// the only point on this path where it can be: after the parse, before
/// anything else walks the result (pounce#472).
fn checked_depth(prob: &NlProblem) -> Result<u32, String> {
    let mut memo = std::collections::HashMap::new();
    let depth = prob
        .con_nonlinear
        .iter()
        .fold(expr_depth(&prob.obj_nonlinear, &mut memo), |acc, c| {
            acc.max(expr_depth(c, &mut memo))
        });
    if depth > MAX_DEPTH {
        return Err(format!(
            "the model nests an expression {depth} levels deep, past the \
             limit of {MAX_DEPTH}. Deeper trees overflow the stack when the model \
             is taped, walked, or freed — a hard crash rather than an exception — \
             so they are refused here. A `.nl` writer that emits `o0` (binary +) \
             chains for a long sum will do this; `o54` (n-ary sum) is one level \
             whatever the term count."
        ));
    }
    Ok(depth)
}

/// Parse an AMPL `.nl` file and return its evaluable [`PyNlProblem`].
///
/// Sibling `.col` / `.row` files (if present) supply variable / constraint
/// names. External (imported) functions are resolved via `AMPLFUNC` exactly
/// as the CLI does.
#[pyfunction]
pub fn read_nl(path: &str) -> PyResult<PyNlProblem> {
    // The parser recurses once per level of nesting in the file, and so
    // does everything downstream of it, so the whole load runs on a stack
    // sized for that rather than on whatever the caller has — the depth is
    // whatever the file says, and there is no way to know it beforehand
    // (pounce#472).
    let path = std::path::Path::new(path);
    let (tnlp, depth) = on_worker_stack(move || {
        let prob = read_nl_file(path)?;
        let depth = checked_depth(&prob)?;
        // `try_new` (not `new`): a model that names an AMPL imported function
        // with no resolvable `$AMPLFUNC` library must raise a catchable Python
        // error, not panic across the pyo3 boundary as an uncatchable
        // PanicException.
        Ok::<_, String>((NlTnlp::try_new(prob)?, depth))
    })
    .map_err(|e| PyValueError::new_err(format!("read_nl: {e}")))?;
    PyNlProblem::from_tnlp(tnlp, "read_nl", depth)
}

/// Parse `.nl` *text* — the same content [`read_nl`] would read off disk —
/// and return its evaluable [`PyNlProblem`] (issue #469).
///
/// This is the no-filesystem route for a frontend that generates `.nl`
/// in memory: no temp file, no cleanup, no `.nl`-writer-dialect surprises
/// from a path that never existed. There are no sibling `.col` / `.row`
/// files to read, so pass `var_names` / `con_names` explicitly if the
/// model has names worth reporting in diagnostics.
///
/// For a frontend that has its own expression DAG, `build_nl_problem` is
/// the better door still: `.nl` cannot spell `atan2`, `min`/`max`, or
/// `erf`, all of which the tape evaluates natively.
#[pyfunction]
#[pyo3(signature = (text, var_names=None, con_names=None))]
pub fn parse_nl_text(
    text: &str,
    var_names: Option<Vec<String>>,
    con_names: Option<Vec<String>>,
) -> PyResult<PyNlProblem> {
    // On a stack sized for the parse, for the reason `read_nl` gives:
    // the text says how deep the recursion goes (pounce#472).
    let (tnlp, depth) = on_worker_stack(move || {
        let mut prob = parse_nl_text_rs(text)?;
        let depth = checked_depth(&prob)?;

        let check = |what: &str, names: &[String], want: usize| -> Result<(), String> {
            if names.len() == want {
                Ok(())
            } else {
                Err(format!(
                    "{what} has length {}, expected {want}",
                    names.len()
                ))
            }
        };
        if let Some(names) = var_names {
            check("var_names", &names, prob.n)?;
            prob.var_names = names;
        }
        if let Some(names) = con_names {
            check("con_names", &names, prob.m)?;
            prob.con_names = names;
        }

        // `try_new` for the same reason `read_nl` uses it: an unresolvable
        // AMPL imported function must be a catchable Python error, not a panic.
        Ok::<_, String>((NlTnlp::try_new(prob)?, depth))
    })
    .map_err(|e| PyValueError::new_err(format!("parse_nl_text: {e}")))?;
    PyNlProblem::from_tnlp(tnlp, "parse_nl_text", depth)
}

#[cfg(test)]
mod tests {
    /// `NlProblem` must stay movable across threads (pounce#477): a
    /// threaded host shares one evaluator across a worker pool, and the
    /// alternative — pyo3's `unsendable` marker — reports a violation as
    /// a `PanicException`, which derives from `BaseException` and so
    /// escapes the host's `except Exception`. A cross-thread call then
    /// reads as a wrong answer rather than an error. Regresses the
    /// moment a `!Send` field (an `Rc`, a non-`Send` trait object)
    /// lands in `PyNlProblem` or in `NlTnlp` beneath it.
    #[test]
    fn py_nl_problem_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<super::PyNlProblem>();
    }
}
