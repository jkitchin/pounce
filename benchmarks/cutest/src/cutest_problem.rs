//! CUTEst problem wrapper implementing POUNCE's
//! [`TNLP`](pounce_nlp::tnlp::TNLP) trait.
//!
//! CUTEst exposes Fortran globals — only ONE problem may be active at a
//! time within a single process. The harness handles this by running each
//! problem in its own subprocess.

#![allow(clippy::too_many_arguments)]

use crate::cutest_ffi::*;
use pounce_common::types::{Index, Number};
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicI32, Ordering};

const CUTEST_INF: f64 = 1e20;
static NEXT_FUNIT: AtomicI32 = AtomicI32::new(55);

fn convert_bound(b: f64) -> f64 {
    if b >= CUTEST_INF {
        f64::INFINITY
    } else if b <= -CUTEST_INF {
        f64::NEG_INFINITY
    } else {
        b
    }
}

/// A CUTEst problem loaded from a compiled `.dylib`/`.so` plus its
/// `OUTSDIF.d` data file.
pub struct CutestProblem {
    pub name: String,
    pub n: usize,
    pub m: usize,
    funit: i32,
    pub x0: Vec<f64>,
    pub x_l: Vec<f64>,
    pub x_u: Vec<f64>,
    pub c_l: Vec<f64>,
    pub c_u: Vec<f64>,
    pub jac_rows: Vec<i32>,
    pub jac_cols: Vec<i32>,
    pub hess_rows: Vec<i32>,
    pub hess_cols: Vec<i32>,
    jac_map: HashMap<(i32, i32), usize>,
    hess_map: HashMap<(i32, i32), usize>,
    nnzj_max: usize,
    nnzh_max: usize,

    // Captured by `finalize_solution`
    pub final_status: Option<SolverReturn>,
    pub final_obj: f64,
    pub final_x: Vec<f64>,
    pub final_g: Vec<f64>,
    pub final_lambda: Vec<f64>,
    pub final_z_l: Vec<f64>,
    pub final_z_u: Vec<f64>,
}

impl CutestProblem {
    /// Load a CUTEst problem from its compiled shared library and `OUTSDIF.d`.
    pub fn load(name: &str, lib_path: &str, outsdif_path: &str) -> Result<Self, String> {
        let lib_cstr =
            CString::new(lib_path).map_err(|e| format!("Invalid lib path: {}", e))?;
        let outsdif_cstr =
            CString::new(outsdif_path).map_err(|e| format!("Invalid OUTSDIF path: {}", e))?;
        let funit = NEXT_FUNIT.fetch_add(1, Ordering::SeqCst);

        unsafe {
            cutest_load_routines(lib_cstr.as_ptr());

            let mut ierr = 0i32;
            fortran_open(&funit, outsdif_cstr.as_ptr(), &mut ierr);
            if ierr != 0 {
                cutest_unload_routines();
                return Err(format!("fortran_open failed with ierr={}", ierr));
            }

            let mut status = 0i32;
            let mut n_i32 = 0i32;
            let mut m_i32 = 0i32;
            cutest_cdimen(&mut status, &funit, &mut n_i32, &mut m_i32);
            if status != 0 {
                fortran_close(&funit, &mut ierr);
                cutest_unload_routines();
                return Err(format!("cutest_cdimen failed with status={}", status));
            }
            let n = n_i32 as usize;
            let m = m_i32 as usize;

            let mut x0 = vec![0.0f64; n];
            let mut x_l = vec![0.0f64; n];
            let mut x_u = vec![0.0f64; n];

            if m > 0 {
                let mut y = vec![0.0f64; m];
                let mut c_l = vec![0.0f64; m];
                let mut c_u = vec![0.0f64; m];
                let mut equatn = vec![false; m];
                let mut linear = vec![false; m];
                let e_order = 0i32;
                let l_order = 0i32;
                let v_order = 0i32;
                let iout = 0i32;
                let io_buffer = 0i32;
                cutest_csetup(
                    &mut status,
                    &funit,
                    &iout,
                    &io_buffer,
                    &mut n_i32,
                    &mut m_i32,
                    x0.as_mut_ptr(),
                    x_l.as_mut_ptr(),
                    x_u.as_mut_ptr(),
                    y.as_mut_ptr(),
                    c_l.as_mut_ptr(),
                    c_u.as_mut_ptr(),
                    equatn.as_mut_ptr(),
                    linear.as_mut_ptr(),
                    &e_order,
                    &l_order,
                    &v_order,
                );
                if status != 0 {
                    cutest_cterminate(&mut status);
                    fortran_close(&funit, &mut ierr);
                    cutest_unload_routines();
                    return Err(format!("cutest_csetup failed with status={}", status));
                }

                for b in x_l.iter_mut() {
                    *b = convert_bound(*b);
                }
                for b in x_u.iter_mut() {
                    *b = convert_bound(*b);
                }
                for b in c_l.iter_mut() {
                    *b = convert_bound(*b);
                }
                for b in c_u.iter_mut() {
                    *b = convert_bound(*b);
                }

                let mut nnzj_max_i32 = 0i32;
                cutest_cdimsj(&mut status, &mut nnzj_max_i32);
                let nnzj_max = nnzj_max_i32 as usize;

                let mut nnzj_i32 = 0i32;
                let mut jvar = vec![0i32; nnzj_max];
                let mut jcon = vec![0i32; nnzj_max];
                cutest_csjp(
                    &mut status,
                    &mut nnzj_i32,
                    &nnzj_max_i32,
                    jvar.as_mut_ptr(),
                    jcon.as_mut_ptr(),
                );
                let nnzj = nnzj_i32 as usize;

                let mut jac_rows = Vec::with_capacity(nnzj);
                let mut jac_cols = Vec::with_capacity(nnzj);
                let mut jac_map = HashMap::with_capacity(nnzj);
                for k in 0..nnzj {
                    let row = jcon[k];
                    let col = jvar[k];
                    jac_rows.push(row);
                    jac_cols.push(col);
                    jac_map.insert((row, col), k);
                }

                let mut nnzh_max_i32 = 0i32;
                cutest_cdimsh(&mut status, &mut nnzh_max_i32);
                let nnzh_max = nnzh_max_i32 as usize;

                let mut nnzh_i32 = 0i32;
                let mut irnh = vec![0i32; nnzh_max];
                let mut icnh = vec![0i32; nnzh_max];
                cutest_cshp(
                    &mut status,
                    &n_i32,
                    &mut nnzh_i32,
                    &nnzh_max_i32,
                    irnh.as_mut_ptr(),
                    icnh.as_mut_ptr(),
                );
                let nnzh = nnzh_i32 as usize;

                let mut hess_rows = Vec::with_capacity(nnzh);
                let mut hess_cols = Vec::with_capacity(nnzh);
                let mut hess_map = HashMap::with_capacity(nnzh);
                for k in 0..nnzh {
                    hess_rows.push(irnh[k]);
                    hess_cols.push(icnh[k]);
                    hess_map.insert((irnh[k], icnh[k]), k);
                }

                Ok(CutestProblem {
                    name: name.to_string(),
                    n,
                    m,
                    funit,
                    x0,
                    x_l,
                    x_u,
                    c_l,
                    c_u,
                    jac_rows,
                    jac_cols,
                    hess_rows,
                    hess_cols,
                    jac_map,
                    hess_map,
                    nnzj_max,
                    nnzh_max,
                    final_status: None,
                    final_obj: f64::NAN,
                    final_x: vec![],
                    final_g: vec![],
                    final_lambda: vec![],
                    final_z_l: vec![],
                    final_z_u: vec![],
                })
            } else {
                let iout = 0i32;
                let io_buffer = 0i32;
                cutest_usetup(
                    &mut status,
                    &funit,
                    &iout,
                    &io_buffer,
                    &mut n_i32,
                    x0.as_mut_ptr(),
                    x_l.as_mut_ptr(),
                    x_u.as_mut_ptr(),
                );
                if status != 0 {
                    cutest_uterminate(&mut status);
                    fortran_close(&funit, &mut ierr);
                    cutest_unload_routines();
                    return Err(format!("cutest_usetup failed with status={}", status));
                }

                for b in x_l.iter_mut() {
                    *b = convert_bound(*b);
                }
                for b in x_u.iter_mut() {
                    *b = convert_bound(*b);
                }

                let mut nnzh_max_i32 = 0i32;
                cutest_udimsh(&mut status, &mut nnzh_max_i32);
                let nnzh_max = nnzh_max_i32 as usize;

                let mut nnzh_i32 = 0i32;
                let mut irnh = vec![0i32; nnzh_max];
                let mut icnh = vec![0i32; nnzh_max];
                cutest_ushp(
                    &mut status,
                    &n_i32,
                    &mut nnzh_i32,
                    &nnzh_max_i32,
                    irnh.as_mut_ptr(),
                    icnh.as_mut_ptr(),
                );
                let nnzh = nnzh_i32 as usize;

                let mut hess_rows = Vec::with_capacity(nnzh);
                let mut hess_cols = Vec::with_capacity(nnzh);
                let mut hess_map = HashMap::with_capacity(nnzh);
                for k in 0..nnzh {
                    hess_rows.push(irnh[k]);
                    hess_cols.push(icnh[k]);
                    hess_map.insert((irnh[k], icnh[k]), k);
                }

                Ok(CutestProblem {
                    name: name.to_string(),
                    n,
                    m: 0,
                    funit,
                    x0,
                    x_l,
                    x_u,
                    c_l: vec![],
                    c_u: vec![],
                    jac_rows: vec![],
                    jac_cols: vec![],
                    hess_rows,
                    hess_cols,
                    jac_map: HashMap::new(),
                    hess_map,
                    nnzj_max: 0,
                    nnzh_max,
                    final_status: None,
                    final_obj: f64::NAN,
                    final_x: vec![],
                    final_g: vec![],
                    final_lambda: vec![],
                    final_z_l: vec![],
                    final_z_u: vec![],
                })
            }
        }
    }

    /// Terminate the CUTEst problem and unload its shared library.
    pub fn cleanup(&self) {
        unsafe {
            let mut status = 0i32;
            if self.m > 0 {
                cutest_cterminate(&mut status);
            } else {
                cutest_uterminate(&mut status);
            }
            let mut ierr = 0i32;
            fortran_close(&self.funit, &mut ierr);
            cutest_unload_routines();
        }
    }

    /// Compute max constraint violation at `x` against the stored bounds.
    pub fn constraint_violation(&self, x: &[f64]) -> f64 {
        if self.m == 0 {
            return 0.0;
        }
        let mut c = vec![0.0f64; self.m];
        let mut status = 0i32;
        let n = self.n as i32;
        let m = self.m as i32;
        unsafe {
            cutest_ccf(&mut status, &n, &m, x.as_ptr(), c.as_mut_ptr());
        }
        let mut max_viol = 0.0f64;
        for i in 0..self.m {
            if c[i] < self.c_l[i] {
                max_viol = max_viol.max(self.c_l[i] - c[i]);
            }
            if c[i] > self.c_u[i] {
                max_viol = max_viol.max(c[i] - self.c_u[i]);
            }
        }
        max_viol
    }
}

impl TNLP for CutestProblem {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as Index,
            m: self.m as Index,
            nnz_jac_g: self.jac_rows.len() as Index,
            nnz_h_lag: self.hess_rows.len() as Index,
            // CUTEst FFI yields 0-based row/col indices.
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        if self.m > 0 {
            b.g_l.copy_from_slice(&self.c_l);
            b.g_u.copy_from_slice(&self.c_u);
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&self.x0);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut status = 0i32;
        let n = self.n as i32;
        let mut f = 0.0f64;
        let grad = false;
        let mut g = vec![0.0f64; self.n];
        unsafe {
            if self.m > 0 {
                cutest_cofg(&mut status, &n, x.as_ptr(), &mut f, g.as_mut_ptr(), &grad);
            } else {
                cutest_uofg(&mut status, &n, x.as_ptr(), &mut f, g.as_mut_ptr(), &grad);
            }
        }
        if status != 0 {
            None
        } else {
            Some(f)
        }
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        let mut status = 0i32;
        let n = self.n as i32;
        let mut f = 0.0f64;
        let grad = true;
        unsafe {
            if self.m > 0 {
                cutest_cofg(
                    &mut status,
                    &n,
                    x.as_ptr(),
                    &mut f,
                    grad_f.as_mut_ptr(),
                    &grad,
                );
            } else {
                cutest_uofg(
                    &mut status,
                    &n,
                    x.as_ptr(),
                    &mut f,
                    grad_f.as_mut_ptr(),
                    &grad,
                );
            }
        }
        status == 0
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        if self.m == 0 {
            return true;
        }
        let mut status = 0i32;
        let n = self.n as i32;
        let m = self.m as i32;
        unsafe {
            cutest_ccf(&mut status, &n, &m, x.as_ptr(), g.as_mut_ptr());
        }
        status == 0
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for k in 0..self.jac_rows.len() {
                    irow[k] = self.jac_rows[k];
                    jcol[k] = self.jac_cols[k];
                }
                true
            }
            SparsityRequest::Values { values } => {
                if self.m == 0 {
                    return true;
                }
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                let mut status = 0i32;
                let n = self.n as i32;
                let m = self.m as i32;
                let lcjac = self.nnzj_max as i32;
                let grad = true;

                let mut c = vec![0.0f64; self.m];
                let mut nnzj = 0i32;
                let mut cjac = vec![0.0f64; self.nnzj_max];
                let mut indvar = vec![0i32; self.nnzj_max];
                let mut indfun = vec![0i32; self.nnzj_max];

                unsafe {
                    cutest_ccfsg(
                        &mut status,
                        &n,
                        &m,
                        x.as_ptr(),
                        c.as_mut_ptr(),
                        &mut nnzj,
                        &lcjac,
                        cjac.as_mut_ptr(),
                        indvar.as_mut_ptr(),
                        indfun.as_mut_ptr(),
                        &grad,
                    );
                }
                if status != 0 {
                    return false;
                }
                for v in values.iter_mut() {
                    *v = 0.0;
                }
                for i in 0..nnzj as usize {
                    let key = (indfun[i], indvar[i]);
                    if let Some(&pos) = self.jac_map.get(&key) {
                        values[pos] = cjac[i];
                    }
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for k in 0..self.hess_rows.len() {
                    irow[k] = self.hess_rows[k];
                    jcol[k] = self.hess_cols[k];
                }
                true
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                let mut status = 0i32;
                let n = self.n as i32;

                for v in values.iter_mut() {
                    *v = 0.0;
                }

                if self.m > 0 {
                    let lambda = lambda.unwrap_or(&[]);
                    let m = self.m as i32;
                    let lh = self.nnzh_max as i32;
                    let mut nnzh = 0i32;
                    let mut h = vec![0.0f64; self.nnzh_max];
                    let mut irnh = vec![0i32; self.nnzh_max];
                    let mut icnh = vec![0i32; self.nnzh_max];
                    unsafe {
                        cutest_cshj(
                            &mut status,
                            &n,
                            &m,
                            x.as_ptr(),
                            &obj_factor,
                            lambda.as_ptr(),
                            &mut nnzh,
                            &lh,
                            h.as_mut_ptr(),
                            irnh.as_mut_ptr(),
                            icnh.as_mut_ptr(),
                        );
                    }
                    if status != 0 {
                        return false;
                    }
                    for i in 0..nnzh as usize {
                        let key = (irnh[i], icnh[i]);
                        if let Some(&pos) = self.hess_map.get(&key) {
                            values[pos] = h[i];
                        }
                    }
                } else {
                    let lh = self.nnzh_max as i32;
                    let mut nnzh = 0i32;
                    let mut h = vec![0.0f64; self.nnzh_max];
                    let mut irnh = vec![0i32; self.nnzh_max];
                    let mut icnh = vec![0i32; self.nnzh_max];
                    unsafe {
                        cutest_ush(
                            &mut status,
                            &n,
                            x.as_ptr(),
                            &mut nnzh,
                            &lh,
                            h.as_mut_ptr(),
                            irnh.as_mut_ptr(),
                            icnh.as_mut_ptr(),
                        );
                    }
                    if status != 0 {
                        return false;
                    }
                    for i in 0..nnzh as usize {
                        let key = (irnh[i], icnh[i]);
                        if let Some(&pos) = self.hess_map.get(&key) {
                            values[pos] = h[i] * obj_factor;
                        }
                    }
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        self.final_status = Some(sol.status);
        self.final_obj = sol.obj_value;
        self.final_x = sol.x.to_vec();
        self.final_g = sol.g.to_vec();
        self.final_lambda = sol.lambda.to_vec();
        self.final_z_l = sol.z_l.to_vec();
        self.final_z_u = sol.z_u.to_vec();
    }
}
