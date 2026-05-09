//! CUTEst problem wrapper implementing ripopt's NlpProblem trait.

use crate::cutest_ffi::*;
use ripopt::NlpProblem;
use std::collections::HashMap;
use std::ffi::CString;

/// A CUTEst problem loaded from a compiled .dylib + OUTSDIF.d pair.
///
/// CUTEst uses global Fortran state — only ONE problem may be active at a time.
#[allow(dead_code)]
pub struct CutestProblem {
    pub name: String,
    pub n: usize,
    pub m: usize,
    funit: i32,
    x0: Vec<f64>,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    c_l: Vec<f64>,
    c_u: Vec<f64>,
    // Fixed sparsity structures (queried once at setup)
    jac_rows: Vec<usize>,
    jac_cols: Vec<usize>,
    hess_rows: Vec<usize>,
    hess_cols: Vec<usize>,
    // Scatter maps: CUTEst (row, col) → position in canonical structure
    jac_map: HashMap<(i32, i32), usize>,
    hess_map: HashMap<(i32, i32), usize>,
    // Max allocation sizes for evaluation buffers
    nnzj_max: usize,
    nnzh_max: usize,
}

const CUTEST_INF: f64 = 1e20;

use std::sync::atomic::{AtomicI32, Ordering};
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

impl CutestProblem {
    /// Load a CUTEst problem from its compiled shared library and data file.
    ///
    /// # Arguments
    /// * `name` - Problem name (e.g., "ROSENBR")
    /// * `lib_path` - Path to the compiled shared library (e.g., "problems/libROSENBR.dylib")
    /// * `outsdif_path` - Path to the OUTSDIF.d data file
    pub fn load(name: &str, lib_path: &str, outsdif_path: &str) -> Result<Self, String> {
        let lib_cstr =
            CString::new(lib_path).map_err(|e| format!("Invalid lib path: {}", e))?;
        let outsdif_cstr =
            CString::new(outsdif_path).map_err(|e| format!("Invalid OUTSDIF path: {}", e))?;

        // Use a unique Fortran unit number for each problem to avoid state reuse
        let funit = NEXT_FUNIT.fetch_add(1, Ordering::SeqCst);

        unsafe {
            // 1. Load the problem's shared library
            cutest_load_routines(lib_cstr.as_ptr());

            // 2. Open the OUTSDIF.d data file
            let mut ierr = 0i32;
            fortran_open(&funit, outsdif_cstr.as_ptr(), &mut ierr);
            if ierr != 0 {
                cutest_unload_routines();
                return Err(format!("fortran_open failed with ierr={}", ierr));
            }

            // 3. Get dimensions
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

            // 4. Setup
            let mut x0 = vec![0.0f64; n];
            let mut x_l = vec![0.0f64; n];
            let mut x_u = vec![0.0f64; n];

            if m > 0 {
                // Constrained setup
                let mut y = vec![0.0f64; m];
                let mut c_l = vec![0.0f64; m];
                let mut c_u = vec![0.0f64; m];
                let mut equatn = vec![false; m];
                let mut linear = vec![false; m];
                let e_order = 0i32;
                let l_order = 0i32;
                let v_order = 0i32;
                let iout = 0i32; // suppress output
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

                // Convert CUTEst bounds to ripopt convention
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

                // 5. Query Jacobian sparsity pattern
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
                    let row = jcon[k]; // constraint index (0-based)
                    let col = jvar[k]; // variable index (0-based)
                    jac_rows.push(row as usize);
                    jac_cols.push(col as usize);
                    jac_map.insert((row, col), k);
                }

                // 6. Query Hessian sparsity pattern
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
                    let row = irnh[k];
                    let col = icnh[k];
                    hess_rows.push(row as usize);
                    hess_cols.push(col as usize);
                    hess_map.insert((row, col), k);
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
                })
            } else {
                // Unconstrained setup
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

                // Convert bounds
                for b in x_l.iter_mut() {
                    *b = convert_bound(*b);
                }
                for b in x_u.iter_mut() {
                    *b = convert_bound(*b);
                }

                // Query Hessian sparsity pattern (unconstrained)
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
                    let row = irnh[k];
                    let col = icnh[k];
                    hess_rows.push(row as usize);
                    hess_cols.push(col as usize);
                    hess_map.insert((row, col), k);
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
                })
            }
        }
    }

    /// Terminate the CUTEst problem and unload the shared library.
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
}

impl NlpProblem for CutestProblem {
    fn num_variables(&self) -> usize {
        self.n
    }

    fn num_constraints(&self) -> usize {
        self.m
    }

    fn bounds(&self, x_l: &mut [f64], x_u: &mut [f64]) {
        x_l.copy_from_slice(&self.x_l);
        x_u.copy_from_slice(&self.x_u);
    }

    fn constraint_bounds(&self, g_l: &mut [f64], g_u: &mut [f64]) {
        g_l.copy_from_slice(&self.c_l);
        g_u.copy_from_slice(&self.c_u);
    }

    fn initial_point(&self, x0: &mut [f64]) {
        x0.copy_from_slice(&self.x0);
    }

    fn objective(&self, x: &[f64], _new_x: bool, obj: &mut f64) -> bool {
        let mut status = 0i32;
        let n = self.n as i32;
        let mut f = 0.0f64;
        let grad = false;
        // g is unused when grad=false, but we need a valid pointer
        let mut g = vec![0.0f64; self.n];
        unsafe {
            if self.m > 0 {
                cutest_cofg(&mut status, &n, x.as_ptr(), &mut f, g.as_mut_ptr(), &grad);
            } else {
                cutest_uofg(&mut status, &n, x.as_ptr(), &mut f, g.as_mut_ptr(), &grad);
            }
        }
        *obj = f;
        true
    }

    fn gradient(&self, x: &[f64], _new_x: bool, grad_out: &mut [f64]) -> bool {
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
                    grad_out.as_mut_ptr(),
                    &grad,
                );
            } else {
                cutest_uofg(
                    &mut status,
                    &n,
                    x.as_ptr(),
                    &mut f,
                    grad_out.as_mut_ptr(),
                    &grad,
                );
            }
        }
        true
    }

    fn constraints(&self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        if self.m == 0 {
            return true;
        }
        let mut status = 0i32;
        let n = self.n as i32;
        let m = self.m as i32;
        unsafe {
            cutest_ccf(&mut status, &n, &m, x.as_ptr(), g.as_mut_ptr());
        }
        true
    }

    fn jacobian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.jac_rows.clone(), self.jac_cols.clone())
    }

    fn jacobian_values(&self, x: &[f64], _new_x: bool, vals: &mut [f64]) -> bool {
        if self.m == 0 {
            return true;
        }
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

        // Initialize output to zero (some structural nonzeros may be numerically zero)
        vals.iter_mut().for_each(|v| *v = 0.0);

        // Scatter CUTEst results into canonical structure
        for i in 0..nnzj as usize {
            let key = (indfun[i], indvar[i]); // (row, col)
            if let Some(&pos) = self.jac_map.get(&key) {
                vals[pos] = cjac[i];
            }
        }
        true
    }

    fn hessian_structure(&self) -> (Vec<usize>, Vec<usize>) {
        (self.hess_rows.clone(), self.hess_cols.clone())
    }

    fn hessian_values(&self, x: &[f64], _new_x: bool, obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> bool {
        let mut status = 0i32;
        let n = self.n as i32;

        // Initialize output to zero
        vals.iter_mut().for_each(|v| *v = 0.0);

        if self.m > 0 {
            // Constrained: use cshj (John function Hessian)
            // H = obj_factor * H_f + sum(lambda_i * H_ci)
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

            for i in 0..nnzh as usize {
                let key = (irnh[i], icnh[i]);
                if let Some(&pos) = self.hess_map.get(&key) {
                    vals[pos] = h[i];
                }
            }
        } else {
            // Unconstrained: use ush, scale by obj_factor
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

            for i in 0..nnzh as usize {
                let key = (irnh[i], icnh[i]);
                if let Some(&pos) = self.hess_map.get(&key) {
                    vals[pos] = h[i] * obj_factor;
                }
            }
        }
        true
    }
}
