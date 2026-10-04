//! gh#983 review fixes. Each test names the branch it reaches.
//!
//! * the CSTR table (item 4): `fs` across `[1e-6, 1e-3]` ends with a small
//!   relative error or a structured warning -- never neither;
//! * benign models are not re-solved (item 2): a start near a stationary point
//!   and a legitimately large objective (`hs71 x 1e8`);
//! * the toll MPCC (items 1 and 5): a gh#884 retry whose certificate is ten
//!   orders better is promoted, and carries no warning from the discarded base
//!   attempt. (`x*y = 0`, the issue's item-4 repro, reaches the retry only
//!   through the second-opinion ladder, so its end-to-end test is in Python:
//!   `python/tests/test_issue_983_review.py`.)

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

fn solve_with(
    tnlp: Rc<RefCell<dyn TNLP>>,
    num: &[(&str, Number)],
    strs: &[(&str, &str)],
) -> (ApplicationReturnStatus, SolveStatistics) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value(
            "print_level",
            std::env::var("POUNCE_TEST_PL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            true,
            false,
        )
        .unwrap();
    for &(k, v) in num {
        app.options_mut()
            .set_numeric_value(k, v, true, false)
            .unwrap();
    }
    for &(k, v) in strs {
        app.options_mut()
            .set_string_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let status = app.optimize_tnlp(tnlp);
    (status, app.statistics())
}

fn has_code(st: &SolveStatistics, code: &str) -> bool {
    st.warnings
        .iter()
        .any(|w| w.starts_with(&format!("{code}:")))
}

// ------------------------------------------------------------------ CSTR ---

/// The issue's item-2 reactor: maximize `fs * (10 cB - 0.02 tau)` subject to
/// the two steady-state balances, `T` capped at 360 (active at the optimum).
struct Cstr {
    fs: Number,
}

const CA0: Number = 2.0;
const K10: Number = 1.0e6;
const E1: Number = 50_000.0;
const K20: Number = 5.0e3;
const E2: Number = 35_000.0;
const RGAS: Number = 8.314;
/// `-obj/fs` at the optimum, from the issue (fs = 1, exact Hessian).
const CSTR_TRUE: Number = 5.352367857245759;

fn arrh(k0: Number, e: Number, t: Number) -> (Number, Number, Number) {
    let k = k0 * (-e / (RGAS * t)).exp();
    let a = e / (RGAS * t * t);
    let dk = k * a;
    let d2k = k * a * a - 2.0 * k * e / (RGAS * t * t * t);
    (k, dk, d2k)
}

impl TNLP for Cstr {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 2,
            nnz_jac_g: 8,
            nnz_h_lag: 10,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, 0.0, 0.1, 300.0]);
        b.x_u.copy_from_slice(&[2.0, 2.0, 60.0, 360.0]);
        b.g_l.fill(0.0);
        b.g_u.fill(0.0);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 1.0, 10.0, 330.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(-self.fs * (10.0 * x[1] - 0.02 * x[2]))
    }
    fn eval_grad_f(&mut self, _x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&[0.0, -10.0 * self.fs, 0.02 * self.fs, 0.0]);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let (ca, cb, tau, t) = (x[0], x[1], x[2], x[3]);
        let (k1, _, _) = arrh(K10, E1, t);
        let (k2, _, _) = arrh(K20, E2, t);
        g[0] = CA0 - ca - tau * k1 * ca;
        g[1] = -cb + tau * (k1 * ca - k2 * cb);
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, (r, c)) in (0..2).flat_map(|r| (0..4).map(move |c| (r, c))).enumerate() {
                    irow[k] = r;
                    jcol[k] = c;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap();
                let (ca, cb, tau, t) = (x[0], x[1], x[2], x[3]);
                let (k1, dk1, _) = arrh(K10, E1, t);
                let (k2, dk2, _) = arrh(K20, E2, t);
                values[..8].copy_from_slice(&[
                    -1.0 - tau * k1,
                    0.0,
                    -k1 * ca,
                    -tau * ca * dk1,
                    tau * k1,
                    -1.0 - tau * k2,
                    k1 * ca - k2 * cb,
                    tau * (ca * dk1 - cb * dk2),
                ]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        _of: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for r in 0..4 {
                    for c in 0..=r {
                        irow[k] = r as i32;
                        jcol[k] = c as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap();
                let l = lambda.unwrap();
                let (ca, cb, tau, t) = (x[0], x[1], x[2], x[3]);
                let (k1, dk1, d2k1) = arrh(K10, E1, t);
                let (k2, dk2, d2k2) = arrh(K20, E2, t);
                // Lower triangle, row-major: (0,0) (1,0) (1,1) (2,0) (2,1)
                // (2,2) (3,0) (3,1) (3,2) (3,3). The objective is linear.
                let mut h = [0.0; 10];
                // g0 = CA0 - ca - tau k1 ca
                h[3] += l[0] * (-k1);
                h[6] += l[0] * (-tau * dk1);
                h[8] += l[0] * (-ca * dk1);
                h[9] += l[0] * (-tau * ca * d2k1);
                // g1 = -cb + tau k1 ca - tau k2 cb
                h[3] += l[1] * k1;
                h[6] += l[1] * tau * dk1;
                h[4] += l[1] * (-k2);
                h[7] += l[1] * (-tau * dk2);
                h[8] += l[1] * (ca * dk1 - cb * dk2);
                h[9] += l[1] * tau * (ca * d2k1 - cb * d2k2);
                values[..10].copy_from_slice(&h);
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn cstr_rel_err(st: &SolveStatistics, fs: Number) -> Number {
    ((-st.final_objective / fs) - CSTR_TRUE).abs() / CSTR_TRUE
}

// ------------------------------------------------------------ toll MPCC ---

/// The issue's item-5 toll-pricing MPCC, `v = (tau, x1..x3, z1..z3, pi)`:
/// maximize `tau * x2` (posed as `min -tau*x2`) subject to the three Wardrop
/// rows, the complementarity products `x_a z_a = 0` and the demand row.
/// `dense` reproduces the issue's `from_jax(..., jac_pattern=dense,
/// hess_pattern=dense lower)` declaration, structural zeros included.
struct Toll {
    dense: bool,
    x: Vec<Number>,
}

const T0: [Number; 3] = [10.0, 12.0, 20.0];
const CAP: [Number; 3] = [1000.0, 1500.0, 800.0];
const DEM: Number = 2500.0;
const VOT: Number = 0.25;

impl Toll {
    fn new(dense: bool) -> Self {
        Toll {
            dense,
            x: Vec::new(),
        }
    }
    /// Sparse Jacobian entries `(row, col)`.
    fn jac_sparse() -> Vec<(i32, i32)> {
        let mut e = Vec::new();
        for a in 0..3 {
            e.push((a as i32, 1 + a as i32)); // x_a
            if a == 1 {
                e.push((1, 0)); // tau
            }
            e.push((a as i32, 4 + a as i32)); // z_a
            e.push((a as i32, 7)); // pi
        }
        for a in 0..3 {
            e.push((3 + a as i32, 1 + a as i32));
            e.push((3 + a as i32, 4 + a as i32));
        }
        for a in 0..3 {
            e.push((6, 1 + a as i32));
        }
        e
    }
    fn jac_entries(&self) -> Vec<(i32, i32)> {
        if self.dense {
            (0..7).flat_map(|r| (0..8).map(move |c| (r, c))).collect()
        } else {
            Self::jac_sparse()
        }
    }
    fn hess_entries(&self) -> Vec<(i32, i32)> {
        if self.dense {
            (0..8).flat_map(|r| (0..=r).map(move |c| (r, c))).collect()
        } else {
            let mut e = vec![(2, 0)];
            for a in 0..3 {
                e.push((1 + a, 1 + a));
                e.push((4 + a, 1 + a));
            }
            e
        }
    }
    fn jac_value(v: &[Number], r: i32, c: i32) -> Number {
        let (r, c) = (r as usize, c as usize);
        match r {
            0..=2 => {
                let a = r;
                if c == 1 + a {
                    T0[a] * 0.15 * 4.0 * v[1 + a].powi(3) / CAP[a].powi(4)
                } else if c == 4 + a || c == 7 {
                    -1.0
                } else if c == 0 && a == 1 {
                    1.0 / VOT
                } else {
                    0.0
                }
            }
            3..=5 => {
                let a = r - 3;
                if c == 1 + a {
                    v[4 + a]
                } else if c == 4 + a {
                    v[1 + a]
                } else {
                    0.0
                }
            }
            _ => {
                if (1..=3).contains(&c) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    fn hess_value(v: &[Number], of: Number, l: &[Number], r: i32, c: i32) -> Number {
        let (r, c) = (r as usize, c as usize);
        let mut h = 0.0;
        if r == 2 && c == 0 {
            h -= of;
        }
        for a in 0..3 {
            if r == 1 + a && c == 1 + a {
                h += l[a] * T0[a] * 0.15 * 12.0 * v[1 + a].powi(2) / CAP[a].powi(4);
            }
            if r == 4 + a && c == 1 + a {
                h += l[3 + a];
            }
        }
        h
    }
}

impl TNLP for Toll {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 8,
            m: 7,
            nnz_jac_g: self.jac_entries().len() as i32,
            nnz_h_lag: self.hess_entries().len() as i32,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(0.0);
        b.x_u
            .copy_from_slice(&[10.0, DEM, DEM, DEM, 100.0, 100.0, 100.0, 100.0]);
        b.g_l.fill(0.0);
        b.g_u.fill(0.0);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let d3 = DEM / 3.0;
        sp.x.copy_from_slice(&[1.0, d3, d3, d3, 1.0, 1.0, 1.0, 15.0]);
        true
    }
    fn eval_f(&mut self, v: &[Number], _n: bool) -> Option<Number> {
        Some(-v[0] * v[2])
    }
    fn eval_grad_f(&mut self, v: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g.fill(0.0);
        g[0] = -v[2];
        g[2] = -v[0];
        true
    }
    fn eval_g(&mut self, v: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for a in 0..3 {
            let toll = if a == 1 { v[0] / VOT } else { 0.0 };
            g[a] = T0[a] * (1.0 + 0.15 * (v[1 + a] / CAP[a]).powi(4)) + toll - v[7] - v[4 + a];
            g[3 + a] = v[1 + a] * v[4 + a];
        }
        g[6] = v[1] + v[2] + v[3] - DEM;
        true
    }
    fn eval_jac_g(&mut self, v: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        let e = self.jac_entries();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in e.iter().enumerate() {
                    irow[k] = r;
                    jcol[k] = c;
                }
            }
            SparsityRequest::Values { values } => {
                let v = v.unwrap();
                for (k, &(r, c)) in e.iter().enumerate() {
                    values[k] = Toll::jac_value(v, r, c);
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        v: Option<&[Number]>,
        _n: bool,
        of: Number,
        l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let e = self.hess_entries();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in e.iter().enumerate() {
                    irow[k] = r;
                    jcol[k] = c;
                }
            }
            SparsityRequest::Values { values } => {
                let (v, l) = (v.unwrap(), l.unwrap());
                for (k, &(r, c)) in e.iter().enumerate() {
                    values[k] = Toll::hess_value(v, of, l, r, c);
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.x = s.x.to_vec();
    }
}

/// gh#983 review item 5: the toll MPCC at `bound_relax_factor = 0`. The base
/// attempt returns `Solved_To_Acceptable_Level` at an unscaled dual
/// infeasibility of `1.03e5` (the gh#884 signature); the retry converges to
/// `1.69e-5` -- ten orders better, above `acceptable_tol` -- and used to be
/// declined on that alone, shipping the runaway. Both Jacobian declarations
/// (the issue's dense `from_jax` patterns and the sparse one) take the same
/// branch.
#[test]
fn toll_mpcc_promotes_a_retry_whose_certificate_dominates() {
    for dense in [true, false] {
        let (status, st) = solve_with(
            Rc::new(RefCell::new(Toll::new(dense))),
            &[("bound_relax_factor", 0.0)],
            &[],
        );
        assert!(st.dual_divergence_signature, "dense {dense}: no signature");
        assert!(
            st.dual_divergence_retry_promoted,
            "dense {dense}: {status:?}, du {:e}",
            st.final_unscaled_dual_inf
        );
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
        assert!(
            st.final_unscaled_kkt_error <= 1e-3,
            "{:e}",
            st.final_unscaled_kkt_error
        );
        // The base attempt's objective was -1735.767543; the promoted one is
        // the same local solution.
        assert!(
            (st.final_objective + 1735.7675).abs() < 1e-3,
            "{}",
            st.final_objective
        );
        // The returned run did not itself show the signature, so nothing of
        // the base attempt's is reported with it.
        assert!(!st.returned_run_dual_divergence_signature);
        assert!(st.warnings.is_empty(), "{:?}", st.warnings);
    }
}

/// The control for the test above: with the default relaxation the base
/// attempt already converges and no retry is spent.
#[test]
fn toll_mpcc_default_relaxation_needs_no_retry() {
    let (status, st) = solve_with(Rc::new(RefCell::new(Toll::new(false))), &[], &[]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(!st.dual_divergence_retry_promoted);
    assert!(
        st.final_unscaled_dual_inf < 1e-3,
        "{:e}",
        st.final_unscaled_dual_inf
    );
}

/// gh#983 review item 1: a second solve on the same application does not
/// inherit the first solve's warnings.
#[test]
fn warnings_do_not_outlive_their_solve() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();
    app.initialize().unwrap();
    // fs = 1e-6 under `none`: the small-objective caveat, no re-solve.
    let _ = app.optimize_tnlp(Rc::new(RefCell::new(Cstr { fs: 1e-6 })));
    assert!(
        has_code(&app.statistics(), "objective_scale_small"),
        "{:?}",
        app.statistics().warnings
    );
    let _ = app.optimize_tnlp(Rc::new(RefCell::new(Cstr { fs: 1.0 })));
    assert!(
        app.statistics().warnings.is_empty(),
        "{:?}",
        app.statistics().warnings
    );
}

// --------------------------------------------------- benign models (item 2) --

/// `min (x-1)^2 + (y-2)^2  s.t.  x + y <= 2`, started a hair from the
/// unconstrained minimizer: `grad f(x0) ~ 2e-7`, `grad f(x*) = (-1, -1)`.
struct NearStationaryStart;

impl TNLP for NearStationaryStart {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-2e19);
        b.x_u.fill(2e19);
        b.g_l[0] = -2e19;
        b.g_u[0] = 2.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 2.0 - 1e-7]);
        true
    }
    fn eval_f(&mut self, v: &[Number], _n: bool) -> Option<Number> {
        Some((v[0] - 1.0).powi(2) + (v[1] - 2.0).powi(2))
    }
    fn eval_grad_f(&mut self, v: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (v[0] - 1.0);
        g[1] = 2.0 * (v[1] - 2.0);
        true
    }
    fn eval_g(&mut self, v: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = v[0] + v[1];
        true
    }
    fn eval_jac_g(&mut self, _v: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[..2].copy_from_slice(&[0, 0]);
                jcol[..2].copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values[..2].copy_from_slice(&[1.0, 1.0]),
        }
        true
    }
    fn eval_h(
        &mut self,
        _v: Option<&[Number]>,
        _n: bool,
        of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[..2].copy_from_slice(&[0, 1]);
                jcol[..2].copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[..2].copy_from_slice(&[2.0 * of, 2.0 * of])
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// gh#983 review item 2(a): the first pass keyed the small-objective branch on
/// the gradient at the start alone and re-solved this model with the
/// objective multiplied by `5e6`. The audit now costs nothing here.
#[test]
fn a_start_near_a_stationary_point_is_not_a_small_objective() {
    let (s_on, on) = solve_with(Rc::new(RefCell::new(NearStationaryStart)), &[], &[]);
    let (s_off, off) = solve_with(
        Rc::new(RefCell::new(NearStationaryStart)),
        &[],
        &[("solve_quality_audit", "no")],
    );
    assert_eq!(s_on, s_off);
    assert!(
        on.start_obj_grad_max < 1e-6,
        "premise: {:e}",
        on.start_obj_grad_max
    );
    assert_eq!(on.iteration_count, off.iteration_count);
    assert_eq!(on.num_obj_evals, off.num_obj_evals);
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
}

// --------------------------------------------------------- CSTR (item 4) --

/// gh#983 review item 4: every `fs` in `[1e-6, 1e-3]` ends with a relative
/// objective error below `1e-6` or carries `objective_scale_small` -- the first
/// pass left `fs = 3e-5 .. 1e-3` with errors up to `1.6e-5` and no signal.
/// The control row of each `fs` (audit off) shows what the audit is for.
#[test]
fn cstr_every_objective_scale_is_fixed_or_flagged() {
    // Measured, audit off (relative error):  1 -> 3.5e-8, 1e-3 -> 4.3e-7,
    // 3e-4 -> 1.5e-6, 1e-4 -> 4.7e-6, 3e-5 -> 1.6e-5, 1e-5 -> 4.7e-5,
    // 3e-6 -> 1.6e-4, 1e-6 -> 4.7e-4. Audit on: 3.0e-8 at every fs <= 1e-3.
    for fs in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6] {
        let (status, st) = solve_with(Rc::new(RefCell::new(Cstr { fs })), &[], &[]);
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "fs {fs:e}");
        let err = cstr_rel_err(&st, fs);
        assert!(
            err < 1e-6 || has_code(&st, "objective_scale_small"),
            "fs {fs:e}: relative error {err:e} and no warning ({:?})",
            st.warnings
        );
        let (_, ctl) = solve_with(
            Rc::new(RefCell::new(Cstr { fs })),
            &[],
            &[("solve_quality_audit", "no")],
        );
        eprintln!(
            "fs {fs:e}: audited {err:.2e}, control {:.2e}",
            cstr_rel_err(&ctl, fs)
        );
    }
    // A well-scaled objective is not re-solved.
    let (_, on) = solve_with(Rc::new(RefCell::new(Cstr { fs: 1.0 })), &[], &[]);
    let (_, off) = solve_with(
        Rc::new(RefCell::new(Cstr { fs: 1.0 })),
        &[],
        &[("solve_quality_audit", "no")],
    );
    assert_eq!(on.iteration_count, off.iteration_count);
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
}

/// The no-re-solve branch: under `nlp_scaling_method = none` the caller owns
/// the scaling, and the small objective is flagged instead of re-solved.
#[test]
fn cstr_small_objective_under_caller_scaling_is_flagged() {
    let (_, st) = solve_with(
        Rc::new(RefCell::new(Cstr { fs: 1e-5 })),
        &[],
        &[("nlp_scaling_method", "none")],
    );
    assert!(has_code(&st, "objective_scale_small"), "{:?}", st.warnings);
}
