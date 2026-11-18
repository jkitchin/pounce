//! gh#1175 — a custom linear solver from C (`linear_solver=custom`), fed
//! per-variable / per-constraint block labels through the metadata setters.
//!
//! Compiles `examples/custom_linear_solver/dense_solver.c` (the reference
//! solver the header documents) into a shared library with the strict C99
//! flags, solves HS071 through the C API with it, and reads back — through
//! the getters that library exports — what pounce handed it:
//!
//! * the solve converges, on the custom solver (it factored);
//! * `set_layout` received `[n_x | n_s | n_c | n_d]`;
//! * `set_metadata` received each integer key mapped to KKT order: an x row
//!   takes its variable's label, a slack and both dual rows their
//!   constraint's, and a key set on one side only reads -1 on the other;
//! * with a fixed variable, its row is gone and the labels still line up;
//! * `custom_linear_solver_options` reaches `create` verbatim;
//! * `linear_solver=custom` without a loadable library refuses the solve
//!   with `Invalid_Option` instead of falling back.
//!
//! HS071's rows are `[x₀x₁x₂x₃ >= 25 (inequality), Σxᵢ² = 40 (equality)]`,
//! so `n_s = n_d = 1` and `n_c = 1`.
//!
//! Compiler as in `c_examples_build_and_run.rs`: `$CC` or `cc`;
//! `POUNCE_C_EXAMPLES_SKIP=1` skips knowingly. Unix only.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_cinterface::*;
use pounce_nlp::ApplicationReturnStatus;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

const N: usize = 4;
const M: usize = 2;

/// The reference solver keeps what it was told in process globals; the
/// tests here take turns.
static SERIAL: Mutex<()> = Mutex::new(());

/// HS071 solution, to ~1e-6.
const X_STAR: [Number; N] = [1.0, 4.742999, 3.821150, 1.379408];

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        1
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(grad, N);
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
        1
    }
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    gout: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(gout, M);
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        1
    }
}

unsafe extern "C" fn ev_jac_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele_jac: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            // Dense 2x4 structure, row-major, 0-based.
            let rows = std::slice::from_raw_parts_mut(i_row, M * N);
            let cols = std::slice::from_raw_parts_mut(j_col, M * N);
            let mut k = 0;
            for i in 0..M {
                for j in 0..N {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let v = std::slice::from_raw_parts_mut(values, M * N);
        v[0] = x[1] * x[2] * x[3];
        v[1] = x[0] * x[2] * x[3];
        v[2] = x[0] * x[1] * x[3];
        v[3] = x[0] * x[1] * x[2];
        v[4] = 2.0 * x[0];
        v[5] = 2.0 * x[1];
        v[6] = 2.0 * x[2];
        v[7] = 2.0 * x[3];
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    const NNZ: usize = N * (N + 1) / 2;
    unsafe {
        if values.is_null() {
            // Dense lower triangle, 0-based.
            let rows = std::slice::from_raw_parts_mut(i_row, NNZ);
            let cols = std::slice::from_raw_parts_mut(j_col, NNZ);
            let mut k = 0;
            for i in 0..N {
                for j in 0..=i {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let lam = std::slice::from_raw_parts(lambda, M);
        let v = std::slice::from_raw_parts_mut(values, NNZ);

        v[0] = obj_factor * (2.0 * x[3]);
        v[1] = obj_factor * x[3];
        v[2] = 0.0;
        v[3] = obj_factor * x[3];
        v[4] = 0.0;
        v[5] = 0.0;
        v[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]);
        v[7] = obj_factor * x[0];
        v[8] = obj_factor * x[0];
        v[9] = 0.0;

        v[1] += lam[0] * (x[2] * x[3]);
        v[3] += lam[0] * (x[1] * x[3]);
        v[4] += lam[0] * (x[0] * x[3]);
        v[6] += lam[0] * (x[1] * x[2]);
        v[7] += lam[0] * (x[0] * x[2]);
        v[8] += lam[0] * (x[0] * x[1]);

        v[0] += lam[1] * 2.0;
        v[2] += lam[1] * 2.0;
        v[5] += lam[1] * 2.0;
        v[9] += lam[1] * 2.0;
        1
    }
}

/// Build the reference solver once; `None` when skipping knowingly.
fn solver_lib() -> Option<&'static PathBuf> {
    static LIB: OnceLock<Option<PathBuf>> = OnceLock::new();
    LIB.get_or_init(|| {
        if std::env::var("POUNCE_C_EXAMPLES_SKIP").as_deref() == Ok("1") {
            return None;
        }
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(if cfg!(target_os = "macos") {
            "libdense_solver_1175.dylib"
        } else {
            "libdense_solver_1175.so"
        });
        let cc = std::env::var("CC")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "cc".into());
        let st = Command::new(&cc)
            .args([
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-O2",
            ])
            .args(["-shared", "-fPIC", "-I"])
            .arg(root.join("include"))
            .arg(root.join("examples/custom_linear_solver/dense_solver.c"))
            .arg("-o")
            .arg(&out)
            .arg("-lm")
            .status()
            .unwrap_or_else(|e| {
                panic!("could not run C compiler `{cc}`: {e} (POUNCE_C_EXAMPLES_SKIP=1 to skip)")
            });
        assert!(st.success(), "compiling dense_solver.c failed");
        Some(out)
    })
    .as_ref()
}

/// The reference solver's introspection getters, from the same loaded
/// image pounce uses (one `dlopen` handle per path per process).
struct Probe(libloading::Library);

impl Probe {
    fn new(lib: &PathBuf) -> Self {
        Probe(unsafe { libloading::Library::new(lib) }.unwrap())
    }
    fn reset(&self) {
        unsafe {
            self.0
                .get::<unsafe extern "C" fn()>(b"dense_solver_reset")
                .unwrap()()
        }
    }
    fn layout(&self) -> [Index; 4] {
        let mut out = [0; 4];
        unsafe {
            self.0
                .get::<unsafe extern "C" fn(*mut Index)>(b"dense_solver_layout")
                .unwrap()(out.as_mut_ptr())
        };
        out
    }
    fn metadata(&self, key: &str) -> Option<Vec<Index>> {
        let k = CString::new(key).unwrap();
        let mut buf = vec![0; 256];
        let len = unsafe {
            self.0
                .get::<unsafe extern "C" fn(*const c_char, *mut Index, Index) -> Index>(
                    b"dense_solver_metadata",
                )
                .unwrap()(k.as_ptr(), buf.as_mut_ptr(), 256)
        };
        (len >= 0).then(|| {
            buf.truncate(len as usize);
            buf
        })
    }
    fn factors(&self) -> i64 {
        unsafe {
            self.0
                .get::<unsafe extern "C" fn() -> std::ffi::c_long>(b"dense_solver_factor_count")
                .unwrap()()
        }
    }
    fn options(&self) -> String {
        unsafe {
            let p = self
                .0
                .get::<unsafe extern "C" fn() -> *const c_char>(b"dense_solver_options")
                .unwrap()();
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn str_opt(p: IpoptProblem, k: &str, v: &str) {
    let (k, v) = (CString::new(k).unwrap(), CString::new(v).unwrap());
    assert_eq!(
        unsafe { AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()) },
        1,
        "{k:?}"
    );
}

/// HS071 with `x_U[0] = x0_upper`: 5.0 is the model, 1.0 fixes x₀ at its
/// lower bound, where the optimum already has it.
fn hs071(x0_upper: Number) -> IpoptProblem {
    let x_l = [1.0; N];
    let mut x_u = [5.0; N];
    x_u[0] = x0_upper;
    let g_l = [25.0, 40.0];
    let g_u = [2.0e19, 40.0];
    let p = unsafe {
        CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            (M * N) as Index,
            (N * (N + 1) / 2) as Index,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        )
    };
    assert!(!p.is_null());
    let k = CString::new("print_level").unwrap();
    assert_eq!(unsafe { AddIpoptIntOption(p, k.as_ptr(), 0) }, 1);
    p
}

fn solve(p: IpoptProblem) -> (Index, [Number; N]) {
    let mut x = [1.0, 5.0, 5.0, 1.0];
    let mut g = [0.0; M];
    let mut obj = 0.0;
    let st = unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (st, x)
}

fn label(p: IpoptProblem) {
    let key = CString::new("block").unwrap();
    let var = [10, 11, 12, 13];
    let con = [20, 21];
    let conly = CString::new("con_only").unwrap();
    unsafe {
        assert_eq!(
            SetPounceIntegerMetaData(p, 0, key.as_ptr(), N as Index, var.as_ptr(), 0),
            0
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 1, key.as_ptr(), M as Index, con.as_ptr(), 0),
            0
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 1, conly.as_ptr(), M as Index, [5, 6].as_ptr(), 0),
            0
        );
    }
}

fn assert_optimum(st: Index, x: [Number; N]) {
    assert_eq!(
        st,
        ApplicationReturnStatus::SolveSucceeded as Index,
        "x = {x:?}"
    );
    for (got, want) in x.iter().zip(X_STAR) {
        assert!((got - want).abs() < 1e-4, "x = {x:?}");
    }
}

#[test]
fn custom_solver_solves_and_receives_kkt_ordered_labels() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lib) = solver_lib() else { return };
    let probe = Probe::new(lib);
    probe.reset();

    let p = hs071(5.0);
    str_opt(p, "linear_solver", "custom");
    str_opt(p, "custom_linear_solver_lib", lib.to_str().unwrap());
    str_opt(p, "custom_linear_solver_options", "pivtol=0.01 verbose");
    label(p);
    let (st, x) = solve(p);
    unsafe { FreeIpoptProblem(p) };

    assert_optimum(st, x);
    assert!(probe.factors() > 0, "the custom solver never factored");
    assert_eq!(probe.options(), "pivtol=0.01 verbose");
    assert_eq!(probe.layout(), [4, 1, 1, 1]);
    // [x0 x1 x2 x3 | s(row 0) | y_c(row 1) | y_d(row 0)]
    assert_eq!(
        probe.metadata("block").unwrap(),
        [10, 11, 12, 13, 20, 21, 20]
    );
    assert_eq!(
        probe.metadata("con_only").unwrap(),
        [-1, -1, -1, -1, 5, 6, 5]
    );
}

#[test]
fn fixed_variable_row_is_dropped_and_labels_stay_aligned() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lib) = solver_lib() else { return };
    let probe = Probe::new(lib);
    probe.reset();

    let p = hs071(1.0); // x₀ fixed at 1
    str_opt(p, "linear_solver", "custom");
    str_opt(p, "custom_linear_solver_lib", lib.to_str().unwrap());
    label(p);
    let (st, x) = solve(p);
    unsafe { FreeIpoptProblem(p) };

    assert_optimum(st, x);
    assert_eq!(probe.layout(), [3, 1, 1, 1]);
    assert_eq!(probe.metadata("block").unwrap(), [11, 12, 13, 20, 21, 20]);
}

#[test]
fn no_metadata_means_no_set_metadata_calls() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lib) = solver_lib() else { return };
    let probe = Probe::new(lib);
    probe.reset();

    let p = hs071(5.0);
    str_opt(p, "linear_solver", "custom");
    str_opt(p, "custom_linear_solver_lib", lib.to_str().unwrap());
    // Set, then cleared: nothing reaches the solver.
    label(p);
    assert_eq!(unsafe { ClearPounceMetaData(p) }, 0);
    let (st, x) = solve(p);
    unsafe { FreeIpoptProblem(p) };

    assert_optimum(st, x);
    assert_eq!(probe.metadata("block"), None);
    assert_eq!(
        probe.layout(),
        [4, 1, 1, 1],
        "set_layout still reports the layout"
    );
}

#[test]
fn custom_without_a_loadable_library_refuses_the_solve() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for lib in [None, Some("/nonexistent/libnope.so")] {
        let p = hs071(5.0);
        str_opt(p, "linear_solver", "custom");
        if let Some(l) = lib {
            str_opt(p, "custom_linear_solver_lib", l);
        }
        let (st, _) = solve(p);
        unsafe { FreeIpoptProblem(p) };
        assert_eq!(
            st,
            ApplicationReturnStatus::InvalidOption as Index,
            "lib = {lib:?}: must refuse, not fall back to another solver"
        );
    }
}

/// Every refusal, in the shared check order, and a refusal changes nothing.
#[test]
fn metadata_setters_refuse_in_check_order() {
    let p = hs071(5.0);
    let key = CString::new("k").unwrap();
    let empty = CString::new("").unwrap();
    let ints = [1 as Index; N];
    let nums = [1.0; N];
    let nan = [1.0, f64::NAN, 1.0, 1.0];
    let s = CString::new("a").unwrap();
    let strs = [s.as_ptr(); N];
    let with_null = [s.as_ptr(), std::ptr::null(), s.as_ptr(), s.as_ptr()];
    let null = std::ptr::null_mut();
    unsafe {
        // error-path: SetPounceIntegerMetaData
        assert_eq!(
            SetPounceIntegerMetaData(null, 0, key.as_ptr(), 4, ints.as_ptr(), 0),
            POUNCE_ERR_NULL
        );
        // NULL handle wins over a bad flag.
        assert_eq!(
            SetPounceIntegerMetaData(null, 0, key.as_ptr(), 4, ints.as_ptr(), 8),
            POUNCE_ERR_NULL
        );
        // flags before target, target before count, count before NULL arrays.
        assert_eq!(
            SetPounceIntegerMetaData(p, 7, std::ptr::null(), 9, std::ptr::null(), 8),
            POUNCE_ERR_FLAGS
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 7, std::ptr::null(), 9, std::ptr::null(), 0),
            POUNCE_ERR_ARGUMENT
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 0, std::ptr::null(), 2, std::ptr::null(), 0),
            POUNCE_ERR_COUNT
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 1, key.as_ptr(), 4, ints.as_ptr(), 0),
            POUNCE_ERR_COUNT
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 0, std::ptr::null(), 4, ints.as_ptr(), 0),
            POUNCE_ERR_NULL
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 0, key.as_ptr(), 4, std::ptr::null(), 0),
            POUNCE_ERR_NULL
        );
        assert_eq!(
            SetPounceIntegerMetaData(p, 0, empty.as_ptr(), 4, ints.as_ptr(), 0),
            POUNCE_ERR_ARGUMENT
        );

        // error-path: SetPounceNumericMetaData
        assert_eq!(
            SetPounceNumericMetaData(p, 0, key.as_ptr(), 3, nums.as_ptr(), 0),
            POUNCE_ERR_COUNT
        );
        assert_eq!(
            SetPounceNumericMetaData(p, 0, key.as_ptr(), 4, nan.as_ptr(), 0),
            POUNCE_ERR_NAN
        );
        assert_eq!(
            SetPounceNumericMetaData(p, 0, key.as_ptr(), 4, nums.as_ptr(), 0),
            POUNCE_OK
        );

        // error-path: SetPounceStringMetaData
        assert_eq!(
            SetPounceStringMetaData(p, 2, key.as_ptr(), 4, strs.as_ptr(), 0),
            POUNCE_ERR_ARGUMENT
        );
        assert_eq!(
            SetPounceStringMetaData(p, 0, key.as_ptr(), 4, with_null.as_ptr(), 0),
            POUNCE_ERR_NULL
        );
        assert_eq!(
            SetPounceStringMetaData(p, 0, key.as_ptr(), 4, strs.as_ptr(), 0),
            POUNCE_OK
        );

        // error-path: ClearPounceMetaData
        assert_eq!(ClearPounceMetaData(null), POUNCE_ERR_NULL);
        assert_eq!(ClearPounceMetaData(p), POUNCE_OK);

        assert_eq!(
            SetPounceIntegerMetaData(p, 1, key.as_ptr(), 2, ints.as_ptr(), 0),
            POUNCE_OK
        );
        FreeIpoptProblem(p);
    }
}

/// Metadata alone — the default linear solver — leaves the solve as it was.
#[test]
fn metadata_without_a_custom_solver_is_harmless() {
    let p = hs071(5.0);
    label(p);
    let (st, x) = solve(p);
    unsafe { FreeIpoptProblem(p) };
    assert_optimum(st, x);
}
