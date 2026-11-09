//! An `.nl` model solved through the C API: the fixture's own evaluator
//! (`pounce_nl::nl_reader::NlTnlp`, the one the CLI solves with) behind the
//! five `extern "C"` callbacks of `CreateIpoptProblem`, its pairs declared
//! with `SetPounceComplementarityPairs`, solved by `IpoptSolve`, reported by
//! `IpoptWriteSolveReport`. Include with
//! `#[path = "common/nl_c_harness.rs"] mod nl_c_harness;`.
//!
//! The C side sees exactly the `.nl` model's bounds, start, sparsity and
//! values — including where the reader puts a row's constant (in the bound,
//! not the body: `2 y1 - 1 - x >= 0` arrives as `2 y1 - x >= 1`). It sees
//! none of the TNLP's optional answers (linearity, nonlinear variables,
//! derivative proofs, scaling suffixes, names): a callback model has no way
//! to say them except `IpoptSetNonlinearVariables` / `SetIpoptProblemScaling`.

#![allow(dead_code)]

use pounce_cinterface::*;
use pounce_nl::nl_reader::{NlTnlp, read_nl_file};
use pounce_nlp::tnlp::{BoundsInfo, SparsityRequest, StartingPoint, TNLP};
use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};

/// The `.nl` evaluator plus the dimensions the trampolines slice with.
pub struct NlHarness {
    pub tnlp: NlTnlp,
    pub n: usize,
    pub m: usize,
    pub nnz_jac: usize,
    pub nnz_hess: usize,
}

unsafe fn harness<'a>(u: *mut c_void) -> &'a mut NlHarness {
    unsafe { &mut *(u as *mut NlHarness) }
}

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    new_x: Bool,
    obj: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let h = harness(u);
        let x = std::slice::from_raw_parts(x, h.n);
        match h.tnlp.eval_f(x, new_x != 0) {
            Some(f) => {
                *obj = f;
                1
            }
            None => 0,
        }
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    new_x: Bool,
    grad: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let h = harness(u);
        let x = std::slice::from_raw_parts(x, h.n);
        let g = std::slice::from_raw_parts_mut(grad, h.n);
        Bool::from(h.tnlp.eval_grad_f(x, new_x != 0, g))
    }
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    new_x: Bool,
    _m: Index,
    gout: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let h = harness(u);
        let x = std::slice::from_raw_parts(x, h.n);
        let g = std::slice::from_raw_parts_mut(gout, h.m);
        Bool::from(h.tnlp.eval_g(x, new_x != 0, g))
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_jac_g(
    _n: Index,
    x: *const Number,
    new_x: Bool,
    _m: Index,
    _nele_jac: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let h = harness(u);
        let nnz = h.nnz_jac;
        let ok = if values.is_null() {
            let irow = std::slice::from_raw_parts_mut(i_row, nnz);
            let jcol = std::slice::from_raw_parts_mut(j_col, nnz);
            h.tnlp
                .eval_jac_g(None, false, SparsityRequest::Structure { irow, jcol })
        } else {
            let x = (!x.is_null()).then(|| std::slice::from_raw_parts(x, h.n));
            let values = std::slice::from_raw_parts_mut(values, nnz);
            h.tnlp
                .eval_jac_g(x, new_x != 0, SparsityRequest::Values { values })
        };
        Bool::from(ok)
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    x: *const Number,
    new_x: Bool,
    obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let h = harness(u);
        let nnz = h.nnz_hess;
        let ok = if values.is_null() {
            let irow = std::slice::from_raw_parts_mut(i_row, nnz);
            let jcol = std::slice::from_raw_parts_mut(j_col, nnz);
            h.tnlp.eval_h(
                None,
                false,
                obj_factor,
                None,
                false,
                SparsityRequest::Structure { irow, jcol },
            )
        } else {
            let x = (!x.is_null()).then(|| std::slice::from_raw_parts(x, h.n));
            let lam = (!lambda.is_null()).then(|| std::slice::from_raw_parts(lambda, h.m));
            let values = std::slice::from_raw_parts_mut(values, nnz);
            h.tnlp.eval_h(
                x,
                new_x != 0,
                obj_factor,
                lam,
                new_lambda != 0,
                SparsityRequest::Values { values },
            )
        };
        Bool::from(ok)
    }
}

/// What one C-API solve of a fixture returned.
pub struct CRun {
    pub status: Index,
    pub x: Vec<Number>,
    pub obj: Number,
    pub iter_count: Index,
    /// The `IpoptWriteSolveReport(path, "full")` JSON.
    pub report: serde_json::Value,
}

/// An option value as the C API takes it: an integer, a real, or a string,
/// decided by how it parses (the CLI's `key=value` goes through
/// `read_from_str`, which coerces by the option's registered type).
fn add_option(prob: IpoptProblem, key: &str, value: &str) {
    let k = CString::new(key).unwrap();
    let ok = unsafe {
        if let Ok(i) = value.parse::<Index>() {
            AddIpoptIntOption(prob, k.as_ptr(), i)
        } else if let Ok(r) = value.parse::<Number>() {
            AddIpoptNumOption(prob, k.as_ptr(), r)
        } else {
            let v = CString::new(value).unwrap();
            AddIpoptStrOption(prob, k.as_ptr(), v.as_ptr())
        }
    };
    assert!(ok != 0, "the C API refused {key}={value}");
}

/// Solve `path` through the C API with `options` (`key=value`, as on the
/// CLI's command line) and write its report into `report_dir`.
pub fn solve(path: &Path, options: &[&str], report_dir: &Path) -> CRun {
    solve_with(path, options, false, report_dir)
}

/// [`solve`], and with `forward_nonlinear` also hand the C API the `.nl`
/// model's nonlinear-variable list (`NlTnlp::get_list_of_nonlinear_variables`)
/// through `IpoptSetNonlinearVariables` — the one piece of the `.nl` path's
/// structure that moves a trajectory here (the limited-memory Hessian is
/// restricted to it), and that a callback model can state.
pub fn solve_with(
    path: &Path,
    options: &[&str],
    forward_nonlinear: bool,
    report_dir: &Path,
) -> CRun {
    let prob = read_nl_file(path).expect("read .nl");
    let pairs: Vec<(Index, Index)> = prob
        .complementarity
        .iter()
        .map(|c| (c.row as Index, c.var as Index))
        .collect();
    let mut tnlp = NlTnlp::try_new(prob).expect("NlTnlp");
    let info = tnlp.get_nlp_info().expect("nlp info");
    let (n, m) = (info.n as usize, info.m as usize);
    assert_eq!(info.index_style as Index, 0, "NlTnlp reports C indexing");
    let (mut x_l, mut x_u, mut g_l, mut g_u) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; m], vec![0.0; m]);
    assert!(tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }));
    let mut x = vec![0.0; n];
    let (mut z_l, mut z_u, mut lam) = (vec![0.0; n], vec![0.0; n], vec![0.0; m]);
    assert!(tnlp.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lam,
    }));
    let mut h = Box::new(NlHarness {
        tnlp,
        n,
        m,
        nnz_jac: info.nnz_jac_g as usize,
        nnz_hess: info.nnz_h_lag as usize,
    });
    let handle = unsafe {
        CreateIpoptProblem(
            n as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            m as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            info.nnz_jac_g,
            info.nnz_h_lag,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        )
    };
    assert!(!handle.is_null(), "CreateIpoptProblem returned NULL");
    add_option(handle, "print_level", "0");
    for o in options {
        let (k, v) = o.split_once('=').expect("key=value");
        add_option(handle, k, v);
    }
    if forward_nonlinear {
        let k = h.tnlp.get_number_of_nonlinear_variables();
        if k >= 0 {
            let mut pos = vec![0 as Index; k as usize];
            assert!(h.tnlp.get_list_of_nonlinear_variables(&mut pos));
            let ok = unsafe { IpoptSetNonlinearVariables(handle, k, pos.as_ptr()) };
            assert!(ok != 0, "IpoptSetNonlinearVariables refused the .nl list");
        }
    }
    let rows: Vec<Index> = pairs.iter().map(|p| p.0).collect();
    let vars: Vec<Index> = pairs.iter().map(|p| p.1).collect();
    let mut bad: Index = 0;
    let rc = unsafe {
        SetPounceComplementarityPairs(
            handle,
            pairs.len() as Index,
            rows.as_ptr(),
            vars.as_ptr(),
            &mut bad,
        )
    };
    assert_eq!(rc, POUNCE_OK, "pairs refused at {bad}");
    let mut g = vec![0.0; m];
    let mut obj = 0.0;
    let (mut mult_g, mut mz_l, mut mz_u) = (vec![0.0; m], vec![0.0; n], vec![0.0; n]);
    let status = unsafe {
        IpoptSolve(
            handle,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            mz_l.as_mut_ptr(),
            mz_u.as_mut_ptr(),
            &mut *h as *mut NlHarness as *mut c_void,
        )
    };
    let iter_count = unsafe { GetIpoptIterCount(handle) };
    std::fs::create_dir_all(report_dir).unwrap();
    let report_path: PathBuf = report_dir.join("c-report.json");
    let p = CString::new(report_path.to_str().unwrap()).unwrap();
    let full = CString::new("full").unwrap();
    let wrote = unsafe { IpoptWriteSolveReport(handle, p.as_ptr(), full.as_ptr()) };
    assert!(wrote != 0, "IpoptWriteSolveReport failed");
    unsafe { FreeIpoptProblem(handle) };
    drop(h);
    let report = serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    CRun {
        status,
        x,
        obj,
        iter_count,
        report,
    }
}
