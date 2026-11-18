//! `linear_solver=custom`: a sparse symmetric linear solver supplied as a
//! shared library through a C vtable (gh#1175).
//!
//! The library exports one symbol,
//!
//! ```c
//! const PounceLinearSolver* pounce_linear_solver(void);
//! ```
//!
//! and pounce loads it from the path in `custom_linear_solver_lib`. The
//! struct is declared in `pounce.h`; [`PounceLinearSolverVTable`] is its Rust
//! mirror and the two must agree field for field. Because selection is an
//! ordinary string option, every frontend reaches it — the CLI, an
//! `ipopt.opt`, the C API's `AddIpoptStrOption`, Python, and the CasADi
//! plugin — and the solver is used wherever pounce would otherwise build a
//! FERAL backend: the interior point, its restoration sub-solve, and the
//! active-set SQP's QP subproblems.
//!
//! # KKT layout and metadata
//!
//! On the interior-point arms the matrix is the augmented system, ordered
//! `[x | slacks | equality duals | inequality duals]`, after fixed variables
//! have been removed and constraints split into equalities and inequalities.
//! A solver that wants to exploit structure needs to know which original
//! variable or constraint each row came from, and that is exactly what a
//! caller cannot reconstruct from outside. So, before `initialize_structure`,
//! pounce calls `set_layout(n_x, n_s, n_c, n_d)` and then `set_metadata` once
//! per integer metadata key the model declared (`SetPounceIntegerMetaData`
//! in C, `var_integer_md` / `con_integer_md` in CasADi), already mapped to
//! KKT order the same way [`crate::application::map_block_structure_to_kkt`]
//! maps block labels: an `x` row takes its variable's value, a slack, an
//! equality dual and an inequality dual take their constraint's. A key given
//! for only one side reads `-1` on the other side's rows.
//!
//! Both calls are skipped, never guessed, when the dimension pounce is about
//! to factor is not `n_x + n_s + n_c + n_d` — the SQP's QP KKT, or a model
//! rewritten by a wrapper that changes the variable count (the ℓ₁ penalty
//! reformulation adds `2·m_eq` variables but forwards metadata of the
//! original length; the length check rejects it there). A label that is off
//! by one row declares a structure the matrix does not have.

use pounce_common::types::{Index, Number};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::{Arc, Mutex};

/// `abi_version` this build understands. Bump on any change to the struct
/// or to the meaning of a callback; a library built against another
/// version is refused at load time, not called with the wrong layout.
pub const POUNCE_LINEAR_SOLVER_ABI_VERSION: u32 = 1;

/// Return codes of `factor` / `solve` (`POUNCE_LINSOL_*` in `pounce.h`).
pub const POUNCE_LINSOL_SUCCESS: c_int = 0;
pub const POUNCE_LINSOL_SINGULAR: c_int = 1;
pub const POUNCE_LINSOL_WRONG_INERTIA: c_int = 2;

/// The symbol a custom-solver library exports.
pub const ENTRY_SYMBOL: &str = "pounce_linear_solver";

/// Rust mirror of `PounceLinearSolver` in `pounce.h`.
#[repr(C)]
pub struct PounceLinearSolverVTable {
    pub struct_size: u32,
    pub abi_version: u32,
    pub name: *const c_char,
    /// Non-zero when `factor` reports the number of negative eigenvalues.
    /// Without it the interior point cannot correct inertia, which on a
    /// nonconvex model costs robustness.
    pub provides_inertia: c_int,
    pub create: Option<unsafe extern "C" fn(options: *const c_char) -> *mut c_void>,
    pub destroy: Option<unsafe extern "C" fn(state: *mut c_void)>,
    pub set_layout: Option<unsafe extern "C" fn(*mut c_void, Index, Index, Index, Index) -> c_int>,
    pub set_metadata:
        Option<unsafe extern "C" fn(*mut c_void, *const c_char, *const Index, Index) -> c_int>,
    pub initialize_structure: Option<
        unsafe extern "C" fn(*mut c_void, Index, Index, *const Index, *const Index) -> c_int,
    >,
    pub factor: Option<
        unsafe extern "C" fn(*mut c_void, *const Number, Index, c_int, Index, *mut Index) -> c_int,
    >,
    pub solve: Option<unsafe extern "C" fn(*mut c_void, Index, Index, *mut Number) -> c_int>,
    pub increase_quality: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
}

/// The KKT layout of the current interior-point solve and its integer
/// metadata in model order, published by the application once the
/// fixed-variable and equality/inequality split are known.
#[derive(Debug, Clone, Default)]
pub struct KktMetadata {
    pub n_x: Index,
    pub n_s: Index,
    pub n_c: Index,
    pub n_d: Index,
    /// key → value per KKT row, length `n_x + n_s + n_c + n_d`.
    pub integers: BTreeMap<String, Vec<Index>>,
}

impl KktMetadata {
    pub fn dim(&self) -> Index {
        self.n_x + self.n_s + self.n_c + self.n_d
    }

    /// Map model-order integer metadata to KKT order. A key whose vector
    /// does not have the model's length on the side that gives it is
    /// dropped, and its name returned, rather than misaligned.
    pub fn from_model(
        cls: &pounce_nlp::tnlp_adapter::BoundClassification,
        var: &BTreeMap<String, Vec<Index>>,
        con: &BTreeMap<String, Vec<Index>>,
    ) -> (Self, Vec<String>) {
        let n_x = cls.x_not_fixed_map.len();
        let n_c = cls.c_map.len();
        let n_d = cls.d_map.len();
        let mut out = KktMetadata {
            n_x: n_x as Index,
            n_s: n_d as Index,
            n_c: n_c as Index,
            n_d: n_d as Index,
            integers: BTreeMap::new(),
        };
        let mut dropped = Vec::new();
        let keys: std::collections::BTreeSet<&String> = var.keys().chain(con.keys()).collect();
        for key in keys {
            let v = var.get(key);
            let c = con.get(key);
            let v_ok = v.is_none_or(|v| v.len() == cls.n_full_x as usize);
            let c_ok = c.is_none_or(|c| c.len() == cls.n_full_g as usize);
            if !(v_ok && c_ok) {
                dropped.push(key.clone());
                continue;
            }
            let vat = |i: Index| v.map_or(-1, |v| v[i as usize]);
            let cat = |i: Index| c.map_or(-1, |c| c[i as usize]);
            let mut vals = Vec::with_capacity(n_x + 2 * n_d + n_c);
            vals.extend(cls.x_not_fixed_map.iter().map(|&i| vat(i)));
            vals.extend(cls.d_map.iter().map(|&i| cat(i))); // slack
            vals.extend(cls.c_map.iter().map(|&i| cat(i))); // equality dual
            vals.extend(cls.d_map.iter().map(|&i| cat(i))); // inequality dual
            out.integers.insert(key.clone(), vals);
        }
        (out, dropped)
    }
}

/// Shared cell the application writes [`KktMetadata`] into and every
/// backend minted for the solve reads at `initialize_structure`. Shared
/// rather than copied because the restoration sub-solve's builder is
/// minted before the layout is known.
pub type KktMetadataCell = Arc<Mutex<Option<KktMetadata>>>;

#[derive(Clone, Copy)]
struct VtPtr(*const PounceLinearSolverVTable);
// SAFETY: the vtable is immutable static data inside a library that is
// never unloaded (see `load`), so sharing the pointer across threads is
// sharing a `&'static`.
unsafe impl Send for VtPtr {}
unsafe impl Sync for VtPtr {}

/// A loaded, validated custom solver: what the algorithm builder carries
/// and mints a fresh [`CustomBackend`] from per backend request.
#[derive(Clone)]
pub struct CustomLinearSolver {
    vt: VtPtr,
    name: String,
    options: CString,
    metadata: Option<KktMetadataCell>,
}

impl std::fmt::Debug for CustomLinearSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomLinearSolver")
            .field("name", &self.name)
            .field("options", &self.options)
            .finish()
    }
}

impl CustomLinearSolver {
    /// Wrap a vtable the caller already holds, after validating it.
    ///
    /// # Safety
    /// `vt` must point to a `PounceLinearSolver` that stays valid, with its
    /// callbacks callable, for the rest of the process.
    pub unsafe fn from_vtable(
        vt: *const PounceLinearSolverVTable,
        options: &str,
    ) -> Result<Self, String> {
        if vt.is_null() {
            return Err("the custom linear solver's vtable is NULL".into());
        }
        // SAFETY: non-null, and valid per this function's contract. Read
        // the two leading u32s before trusting the rest of the layout.
        let (size, abi) = unsafe { ((*vt).struct_size, (*vt).abi_version) };
        if abi != POUNCE_LINEAR_SOLVER_ABI_VERSION {
            return Err(format!(
                "the custom linear solver was built for PounceLinearSolver ABI {abi}; \
                 this pounce speaks {POUNCE_LINEAR_SOLVER_ABI_VERSION}"
            ));
        }
        if (size as usize) < std::mem::size_of::<PounceLinearSolverVTable>() {
            return Err(format!(
                "the custom linear solver's struct_size is {size}, smaller than the \
                 {} bytes of PounceLinearSolver",
                std::mem::size_of::<PounceLinearSolverVTable>()
            ));
        }
        // SAFETY: size and version checked above.
        let t = unsafe { &*vt };
        let missing: Vec<&str> = [
            ("create", t.create.is_none()),
            ("destroy", t.destroy.is_none()),
            ("initialize_structure", t.initialize_structure.is_none()),
            ("factor", t.factor.is_none()),
            ("solve", t.solve.is_none()),
        ]
        .iter()
        .filter(|(_, m)| *m)
        .map(|(n, _)| *n)
        .collect();
        if !missing.is_empty() {
            return Err(format!(
                "the custom linear solver leaves required callbacks NULL: {}",
                missing.join(", ")
            ));
        }
        let name = if t.name.is_null() {
            "custom".to_string()
        } else {
            // SAFETY: a non-null name is a NUL-terminated string per the
            // header's contract.
            unsafe { CStr::from_ptr(t.name) }
                .to_string_lossy()
                .into_owned()
        };
        let options = CString::new(options)
            .map_err(|_| "custom_linear_solver_options contains a NUL byte".to_string())?;
        Ok(CustomLinearSolver {
            vt: VtPtr(vt),
            name,
            options,
            metadata: None,
        })
    }

    /// The solver's self-reported name, for the banner and messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Attach the cell the application publishes KKT metadata through.
    pub fn with_metadata(mut self, cell: KktMetadataCell) -> Self {
        self.metadata = Some(cell);
        self
    }

    /// The same solver with no metadata: for arms whose KKT is not the
    /// interior point's augmented system (the SQP's QP subproblems), where
    /// a dimension that happens to match would still be a different layout.
    pub fn without_metadata(&self) -> Self {
        CustomLinearSolver {
            metadata: None,
            ..self.clone()
        }
    }

    /// A fresh backend instance (one `create` call).
    pub fn make_backend(&self) -> Box<dyn SparseSymLinearSolverInterface> {
        // SAFETY: validated in `from_vtable`; `create` is non-null.
        let t = unsafe { &*self.vt.0 };
        let state = match t.create {
            // SAFETY: the options pointer is a live NUL-terminated string.
            Some(f) => unsafe { f(self.options.as_ptr()) },
            None => std::ptr::null_mut(),
        };
        Box::new(CustomBackend {
            vt: self.vt,
            state,
            name: self.name.clone(),
            metadata: self.metadata.clone(),
            values: Vec::new(),
            neg_evals: 0,
        })
    }
}

/// Load the solver in `path`. Libraries are cached by path and never
/// unloaded: a solver's state may be live in an `IpoptProblem` the caller
/// still holds, and `dlclose` under it is undefined behaviour.
#[cfg(any(unix, windows))]
pub fn load(path: &str, options: &str) -> Result<CustomLinearSolver, String> {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static CACHE: OnceLock<Mutex<HashMap<String, VtPtr>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| "custom linear solver cache poisoned".to_string())?;
    let vt = match cache.get(path) {
        Some(vt) => *vt,
        None => {
            // SAFETY: loading a library runs its initializers; that is the
            // point of the option, and the path is the user's choice.
            let lib = unsafe { libloading::Library::new(path) }
                .map_err(|e| format!("cannot load custom_linear_solver_lib '{path}': {e}"))?;
            let lib: &'static libloading::Library = Box::leak(Box::new(lib));
            // SAFETY: the symbol's type is fixed by the header.
            let entry: libloading::Symbol<
                '_,
                unsafe extern "C" fn() -> *const PounceLinearSolverVTable,
            > = unsafe { lib.get(ENTRY_SYMBOL.as_bytes()) }
                .map_err(|e| format!("'{path}' does not export `{ENTRY_SYMBOL}`: {e}"))?;
            // SAFETY: calling the documented entry point.
            let vt = VtPtr(unsafe { entry() });
            cache.insert(path.to_string(), vt);
            vt
        }
    };
    // SAFETY: the library is leaked, so the vtable lives for the process.
    unsafe { CustomLinearSolver::from_vtable(vt.0, options) }
}

#[cfg(not(any(unix, windows)))]
pub fn load(path: &str, _options: &str) -> Result<CustomLinearSolver, String> {
    Err(format!(
        "cannot load custom_linear_solver_lib '{path}': loading shared libraries is \
         not supported on this target"
    ))
}

/// One solver instance behind the vtable.
pub struct CustomBackend {
    vt: VtPtr,
    state: *mut c_void,
    name: String,
    metadata: Option<KktMetadataCell>,
    values: Vec<Number>,
    neg_evals: Index,
}

impl Drop for CustomBackend {
    fn drop(&mut self) {
        // SAFETY: validated vtable; `destroy` is non-null; state came from
        // `create` and is destroyed once.
        let t = unsafe { &*self.vt.0 };
        if !self.state.is_null()
            && let Some(f) = t.destroy
        {
            unsafe { f(self.state) };
        }
    }
}

impl CustomBackend {
    fn t(&self) -> &PounceLinearSolverVTable {
        // SAFETY: validated in `from_vtable`, lives for the process.
        unsafe { &*self.vt.0 }
    }

    fn map_status(&self, what: &str, rc: c_int) -> ESymSolverStatus {
        match rc {
            POUNCE_LINSOL_SUCCESS => ESymSolverStatus::Success,
            POUNCE_LINSOL_SINGULAR => ESymSolverStatus::Singular,
            POUNCE_LINSOL_WRONG_INERTIA => ESymSolverStatus::WrongInertia,
            other => {
                tracing::warn!(
                    target: "pounce::linsol",
                    solver = %self.name,
                    "custom linear solver: {what} returned {other}"
                );
                ESymSolverStatus::FatalError
            }
        }
    }

    /// Hand the layout and KKT-ordered metadata to the solver, if the
    /// matrix about to be analysed is the system they describe.
    fn publish_metadata(&mut self, dim: Index) {
        let Some(cell) = &self.metadata else { return };
        let Some(meta) = cell.lock().ok().and_then(|g| g.clone()) else {
            return;
        };
        if meta.dim() != dim {
            return;
        }
        let t = self.t();
        if let Some(f) = t.set_layout {
            // SAFETY: live state, plain integers.
            unsafe { f(self.state, meta.n_x, meta.n_s, meta.n_c, meta.n_d) };
        }
        if let Some(f) = t.set_metadata {
            for (key, vals) in &meta.integers {
                let Ok(k) = CString::new(key.as_str()) else {
                    continue;
                };
                // SAFETY: both pointers live for the call; len is theirs.
                unsafe { f(self.state, k.as_ptr(), vals.as_ptr(), vals.len() as Index) };
            }
        }
    }
}

impl SparseSymLinearSolverInterface for CustomBackend {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        if self.state.is_null() {
            tracing::warn!(
                target: "pounce::linsol",
                solver = %self.name,
                "custom linear solver: create returned NULL"
            );
            return ESymSolverStatus::FatalError;
        }
        self.publish_metadata(dim);
        self.values = vec![0.0; nonzeros.max(0) as usize];
        let Some(f) = self.t().initialize_structure else {
            return ESymSolverStatus::FatalError;
        };
        // SAFETY: `ia`/`ja` hold `nonzeros` entries each (TSymLinearSolver's
        // triplet), live for the call.
        let rc = unsafe { f(self.state, dim, nonzeros, ia.as_ptr(), ja.as_ptr()) };
        self.map_status("initialize_structure", rc)
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        &mut self.values
    }

    fn multi_solve(
        &mut self,
        new_matrix: bool,
        _ia: &[Index],
        _ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        if new_matrix {
            let Some(f) = self.t().factor else {
                return ESymSolverStatus::FatalError;
            };
            let mut neg: Index = 0;
            // SAFETY: `values` has the `nonzeros` entries declared at
            // `initialize_structure`; `neg` is a live out-parameter.
            let rc = unsafe {
                f(
                    self.state,
                    self.values.as_ptr(),
                    self.values.len() as Index,
                    c_int::from(check_neg_evals),
                    number_of_neg_evals,
                    &mut neg,
                )
            };
            self.neg_evals = neg;
            let s = self.map_status("factor", rc);
            if s != ESymSolverStatus::Success {
                return s;
            }
            if check_neg_evals && self.t().provides_inertia != 0 && neg != number_of_neg_evals {
                return ESymSolverStatus::WrongInertia;
            }
        }
        let Some(f) = self.t().solve else {
            return ESymSolverStatus::FatalError;
        };
        let dim = if nrhs > 0 {
            (rhs_vals.len() / nrhs as usize) as Index
        } else {
            0
        };
        // SAFETY: `rhs_vals` holds `nrhs` column-major vectors of `dim`.
        let rc = unsafe { f(self.state, dim, nrhs, rhs_vals.as_mut_ptr()) };
        self.map_status("solve", rc)
    }

    fn number_of_neg_evals(&self) -> Index {
        self.neg_evals
    }

    fn increase_quality(&mut self) -> bool {
        match self.t().increase_quality {
            // SAFETY: live state.
            Some(f) => (unsafe { f(self.state) }) != 0,
            None => false,
        }
    }

    fn provides_inertia(&self) -> bool {
        self.t().provides_inertia != 0
    }

    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::tnlp_adapter::BoundClassification;

    /// 4 variables, x₁ fixed; rows 0 and 2 inequalities, 1 and 3 equalities
    /// — interleaved, so a mapping that assumed eq-first model order fails.
    fn cls() -> BoundClassification {
        BoundClassification {
            n_full_x: 4,
            n_full_g: 4,
            n_x_fixed: 1,
            x_not_fixed_map: vec![0, 2, 3],
            x_fixed_map: vec![1],
            x_fixed_vals: vec![0.0],
            full_to_var: vec![0, -1, 1, 2],
            x_l_map: vec![],
            x_u_map: vec![],
            n_c: 2,
            c_map: vec![1, 3],
            n_d: 2,
            d_map: vec![0, 2],
            d_l_map: vec![],
            d_u_map: vec![],
            full_to_c: vec![-1, 0, -1, 1],
            full_to_d: vec![0, -1, 1, -1],
        }
    }

    #[test]
    fn maps_to_kkt_order_and_drops_misfit_keys() {
        let var = BTreeMap::from([
            ("blk".to_string(), vec![10, 11, 12, 13]),
            ("short".to_string(), vec![1, 2]),
        ]);
        let con = BTreeMap::from([
            ("blk".to_string(), vec![20, 21, 22, 23]),
            ("con_only".to_string(), vec![5, 6, 7, 8]),
        ]);
        let (m, dropped) = KktMetadata::from_model(&cls(), &var, &con);
        assert_eq!((m.n_x, m.n_s, m.n_c, m.n_d, m.dim()), (3, 2, 2, 2, 9));
        // [x0 x2 x3 | s(r0) s(r2) | y_c(r1) y_c(r3) | y_d(r0) y_d(r2)]
        assert_eq!(m.integers["blk"], [10, 12, 13, 20, 22, 21, 23, 20, 22]);
        assert_eq!(m.integers["con_only"], [-1, -1, -1, 5, 7, 6, 8, 5, 7]);
        assert_eq!(dropped, ["short"]);
        assert!(!m.integers.contains_key("short"));
    }
}
