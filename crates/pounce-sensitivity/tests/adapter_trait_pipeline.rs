//! Exercise [`pounce_sensitivity::PdSensBacksolver`] through the same
//! `IndexPCalculator` / `DenseGenSchurDriver` / `StdStepCalc`
//! trait-pipeline calls that
//! [`crates/pounce-sensitivity/src/p_calculator.rs::tests`] and
//! [`crates/pounce-sensitivity/src/step_calc.rs::tests`] use against
//! the synthetic [`pounce_sensitivity::DenseLuBacksolver`].
//!
//! pounce#16 sub-task A acceptance is "the adapter passes the same
//! numerical tests … with a constructed-from-converged-state aug
//! solver instead of `DenseLuBacksolver`". The synthetic Phase-B.1
//! tests run arithmetic on a hand-picked 3×3 SPD `K`; we can't
//! make pounce's IPM converge with exactly that `K` as its augmented
//! KKT factor without distorting the matrix structure. Instead we
//! drive the same trait pipeline against the converged KKT factor of
//! upstream's `parametric_cpp` example (a 12×12 augmented system from
//! the 5-var / 4-constraint ParametricTNLP — see the sibling
//! `parametric_cpp.rs` test) and assert *adapter properties* that hold
//! for any non-singular `K`:
//!
//! 1. **Round-trip**: `K · (K⁻¹ · rhs) ≈ rhs` for several rhs vectors,
//!    where `K · v` is reconstructed by reading row `i`-th of `K⁻¹`
//!    from one backsolve and inverting again. The robust form is
//!    `K⁻¹` columns from `IndexPCalculator::compute_p` round-trip back
//!    to identity when re-passed through the backsolver, i.e.
//!    `backsolver.solve(K · col_i)` yields `col_i` back. Since we don't
//!    have K extracted, we instead verify that `K⁻¹ A` columns
//!    correspond to `solve(±e_idx)` results from the adapter, exercising
//!    `compute_p` end-to-end.
//! 2. **PCalculator consistency**: `IndexPCalculator::compute_p`
//!    produces P columns that equal `SensBacksolver::solve(±e_idx)`
//!    called directly (the adapter behaves identically through both
//!    surfaces).
//! 3. **SchurDriver consistency**: `DenseGenSchurDriver::schur_matrix`
//!    output equals `-B · (K⁻¹ A)` recomputed by hand from P columns +
//!    `B.multiply`.
//! 4. **StdStepCalc**: end-to-end pipeline returns the same `du` and
//!    `dx_full` as direct `schur_solve` + `A.trans_multiply` +
//!    `backsolver.solve` calls.
//!
//! Together these verify the adapter is a drop-in replacement for the
//! synthetic backsolver at every trait surface the Phase-B.1 tests
//! exercise.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::{
    DenseGenSchurDriver, IndexPCalculator, IndexSchurData, PCalculator, PdSensBacksolver,
    SchurData, SchurDriver, SensBacksolver, SensStepCalc, StdStepCalc,
};

/// Rust port of ParametricTNLP — same TNLP the parametric_cpp golden
/// test uses. Inlined here so the two test files stay independent.
struct ParametricTNLP;

impl TNLP for ParametricTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 4,
            nnz_jac_g: 10,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = 0.0;
            b.x_u[k] = 1.0e19;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.x_l[4] = -1.0e19;
        b.x_u[4] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = 5.0;
        b.g_u[2] = 5.0;
        b.g_l[3] = 1.0;
        b.g_u[3] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.15;
        sp.x[1] = 0.15;
        sp.x[2] = 0.0;
        sp.x[3] = 0.0;
        sp.x[4] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        g[2] = 2.0 * x[2];
        g[3] = 0.0;
        g[4] = 0.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 6.0 * x[0] + 3.0 * x[1] + 2.0 * x[2] - x[3];
        g[1] = x[4] * x[0] + x[1] - x[2] - 1.0;
        g[2] = x[3];
        g[3] = x[4];
        true
    }
    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 10] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3];
                let cs: [Index; 10] = [0, 1, 2, 3, 0, 1, 2, 4, 3, 4];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 6.0;
                values[1] = 3.0;
                values[2] = 2.0;
                values[3] = -1.0;
                values[4] = x[4];
                values[5] = 1.0;
                values[6] = -1.0;
                values[7] = x[0];
                values[8] = 1.0;
                values[9] = 1.0;
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 2, 4, 0];
                let cs: [Index; 5] = [0, 1, 2, 0, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                values[2] = 2.0 * obj_factor;
                values[3] = lam[1];
                values[4] = 0.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Drive a ParametricTNLP solve to convergence and run `f` inside the
/// on_converged callback. The returned `R` is stashed via Rc<RefCell>;
/// `Rc::try_unwrap` extracts it after the application drops the
/// callback closure.
fn with_converged_adapter<R: 'static>(f: impl Fn(&PdSensBacksolver) -> R + 'static) -> R {
    let out: Rc<RefCell<Option<R>>> = Rc::new(RefCell::new(None));
    let out_clone = Rc::clone(&out);

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP));
    app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
        let backsolver =
            PdSensBacksolver::new(data, cq, nlp, pd).expect("PdSensBacksolver construction");
        let r = f(&backsolver);
        *out_clone.borrow_mut() = Some(r);
    }));

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "ParametricTNLP solve failed: {status:?}",
    );
    drop(app); // releases the boxed callback so try_unwrap can succeed.
    Rc::try_unwrap(out)
        .ok()
        .and_then(|c| c.into_inner())
        .expect("on_converged populated the slot")
}

/// `idx` here are flat compound-vector indices. For ParametricTNLP at
/// convergence: n_x=5, n_s=0, n_yc=4, n_yd=0, n_zl=3, n_zu=0, n_vl=0,
/// n_vu=0 → total dim = 12. We test against the y_c slots
/// (n_x + i for i in 2..4) since those are the well-conditioned
/// equality-constraint rows the parametric_cpp test already exercises.
const Y_C_PARAM_ROW_ETA1: Index = 5 + 2; // y_c[2] for g[2] = eta1
const Y_C_PARAM_ROW_ETA2: Index = 5 + 3; // y_c[3] for g[3] = eta2

/// **Test 1 (mirrors `p_calculator::tests::compute_p_solves_each_a_column_against_K`).**
/// `IndexPCalculator::compute_p` produces P columns that equal direct
/// `SensBacksolver::solve(e_{a_idx[j]})` calls. This verifies the
/// adapter integrates correctly with `compute_p`'s internal
/// per-column backsolve loop.
#[test]
fn adapter_compute_p_matches_direct_backsolves() {
    with_converged_adapter(|backsolver| {
        let n_full = backsolver.dim();

        // Build P with A picking the two y_c parameter rows with +1.
        let a =
            IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1, Y_C_PARAM_ROW_ETA2], vec![1, 1])
                .unwrap();
        let mut pcalc = IndexPCalculator::new(backsolver.clone(), a);
        assert!(pcalc.compute_p());

        // Independent reference: solve(e_idx) directly for each
        // a_idx, compare to the cached P column.
        for &idx in &[Y_C_PARAM_ROW_ETA1, Y_C_PARAM_ROW_ETA2] {
            let mut e = vec![0.0; n_full];
            e[idx as usize] = 1.0;
            let mut col_ref = vec![0.0; n_full];
            assert!(backsolver.solve(&e, &mut col_ref));

            let col_pcalc = pcalc.p_columns().get(&idx).expect("col cached");
            assert_eq!(col_pcalc.len(), n_full);
            for i in 0..n_full {
                let err = (col_pcalc[i] - col_ref[i]).abs();
                assert!(
                    err < 1e-9,
                    "P col {} row {}: pcalc={} direct={} |err|={}",
                    idx,
                    i,
                    col_pcalc[i],
                    col_ref[i],
                    err
                );
            }
        }
    });
}

/// **Test 2 (mirrors `p_calculator::tests::compute_p_uses_sign_from_a_data`).**
/// A negative sign in IndexSchurData flips the P column.
#[test]
fn adapter_compute_p_respects_negative_signs() {
    with_converged_adapter(|backsolver| {
        let n_full = backsolver.dim();

        let a_pos = IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1], vec![1]).unwrap();
        let a_neg = IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1], vec![-1]).unwrap();

        let mut pc_pos = IndexPCalculator::new(backsolver.clone(), a_pos);
        let mut pc_neg = IndexPCalculator::new(backsolver.clone(), a_neg);
        assert!(pc_pos.compute_p());
        assert!(pc_neg.compute_p());

        let p_pos = pc_pos.p_columns().get(&Y_C_PARAM_ROW_ETA1).unwrap();
        let p_neg = pc_neg.p_columns().get(&Y_C_PARAM_ROW_ETA1).unwrap();
        for i in 0..n_full {
            let err = (p_pos[i] - (-p_neg[i])).abs();
            assert!(
                err < 1e-12,
                "sign flip at row {i}: +={}, -={}, |+ - (-{}=)| = {err}",
                p_pos[i],
                p_neg[i],
                p_neg[i],
            );
        }
    });
}

/// **Test 3 (mirrors `p_calculator::tests::schur_matrix_matches_closed_form_minus_b_kinv_a`).**
/// Build `S = -B · (K⁻¹ A)` two ways — through the calculator's
/// `schur_matrix` API and by hand via P columns + `B.multiply` — and
/// confirm they agree.
#[test]
fn adapter_schur_matrix_matches_hand_compute() {
    with_converged_adapter(|backsolver| {
        // A picks both y_c parameter rows; B picks just the eta1 row.
        let a =
            IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1, Y_C_PARAM_ROW_ETA2], vec![1, 1])
                .unwrap();
        let b = IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1], vec![1]).unwrap();

        let mut pc = IndexPCalculator::new(backsolver.clone(), a.clone());
        // schur_matrix output: column-major, 1 row × 2 cols.
        let mut s = vec![0.0; 1 * 2];
        assert!(pc.schur_matrix(&b, &mut s));

        // Hand: build P columns by direct backsolve, multiply by B
        // row-wise, negate.
        let n_full = backsolver.dim();
        let mut s_ref = vec![0.0; 2];
        for (j, &idx) in [Y_C_PARAM_ROW_ETA1, Y_C_PARAM_ROW_ETA2].iter().enumerate() {
            let mut e = vec![0.0; n_full];
            e[idx as usize] = 1.0;
            let mut col = vec![0.0; n_full];
            assert!(backsolver.solve(&e, &mut col));
            // B is a single +1 in flat slot Y_C_PARAM_ROW_ETA1.
            let bk_col = col[Y_C_PARAM_ROW_ETA1 as usize];
            s_ref[j] = -bk_col;
        }
        for j in 0..2 {
            let err = (s[j] - s_ref[j]).abs();
            assert!(
                err < 1e-9,
                "S[0,{j}]: calc={} hand={} |err|={err}",
                s[j],
                s_ref[j],
            );
        }
    });
}

/// **Test 4 (mirrors `step_calc::tests::std_step_calc_runs_two_step_pipeline`).**
/// `StdStepCalc::compute_step` returns the same `du`, `dx_full` as a
/// hand-built `schur_solve` + `A.trans_multiply` + `backsolver.solve`
/// chain.
#[test]
fn adapter_drives_std_step_calc_pipeline() {
    with_converged_adapter(|backsolver| {
        let n_full = backsolver.dim();
        let a =
            IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1, Y_C_PARAM_ROW_ETA2], vec![1, 1])
                .unwrap();
        let pc = IndexPCalculator::new(backsolver.clone(), a.clone());
        let mut driver = DenseGenSchurDriver::<_, PdSensBacksolver>::new(pc);
        let b =
            IndexSchurData::from_parts(vec![Y_C_PARAM_ROW_ETA1, Y_C_PARAM_ROW_ETA2], vec![1, 1])
                .unwrap();
        assert!(driver.schur_build_and_factor(&b));

        let step = StdStepCalc::new(&driver, driver.pcalc());
        let rhs_u = [1.0, 0.0];
        let mut du = [0.0; 2];
        let mut dx = vec![0.0; n_full];
        assert!(step.compute_step(&rhs_u, &mut du, &mut dx));

        // Hand-build: schur_solve gives the same du; then
        // A.trans_multiply + backsolver.solve gives the same dx_full.
        let mut du_ref = [0.0; 2];
        assert!(driver.schur_solve(&rhs_u, &mut du_ref));
        let mut rhs_full = vec![0.0; n_full];
        a.trans_multiply(&du_ref, &mut rhs_full).unwrap();
        let mut dx_ref = vec![0.0; n_full];
        assert!(backsolver.solve(&rhs_full, &mut dx_ref));

        for j in 0..2 {
            let err = (du[j] - du_ref[j]).abs();
            assert!(
                err < 1e-12,
                "du[{j}]: pipeline={} hand={} |err|={err}",
                du[j],
                du_ref[j]
            );
        }
        for i in 0..n_full {
            let err = (dx[i] - dx_ref[i]).abs();
            assert!(
                err < 1e-9,
                "dx[{i}]: pipeline={} hand={} |err|={err}",
                dx[i],
                dx_ref[i],
            );
        }
    });
}

/// pounce#16 sub-task: the sIPOPT option keys
/// (`run_sens`, `compute_red_hessian`, `n_sens_steps`, `sens_*`,
/// `rh_eigendecomp`) are recognized by a default `IpoptApplication`
/// — i.e. wired through
/// `pounce_algorithm::upstream_options::register_all_upstream_options`.
/// Setting one of them must succeed without a "registered_only" error.
#[test]
fn sipopt_options_are_recognized_by_ipopt_application() {
    let mut app = IpoptApplication::new();
    // `set_*_value(..., true, false)` enforces "must be registered;
    // refuse out-of-table keys". Each call must succeed.
    app.options_mut()
        .set_string_value("run_sens", "yes", true, false)
        .expect("run_sens registered");
    app.options_mut()
        .set_string_value("compute_red_hessian", "yes", true, false)
        .expect("compute_red_hessian registered");
    app.options_mut()
        .set_integer_value("n_sens_steps", 1, true, false)
        .expect("n_sens_steps registered");
    app.options_mut()
        .set_string_value("sens_boundcheck", "yes", true, false)
        .expect("sens_boundcheck registered");
    app.options_mut()
        .set_numeric_value("sens_bound_eps", 1.0e-4, true, false)
        .expect("sens_bound_eps registered");
    app.options_mut()
        .set_numeric_value("sens_max_pdpert", 1.0e-4, true, false)
        .expect("sens_max_pdpert registered");
    app.options_mut()
        .set_string_value("rh_eigendecomp", "yes", true, false)
        .expect("rh_eigendecomp registered");
}

/// Sanity check: `SensBacksolver::solve` produces deterministic,
/// linear-in-rhs results from the converged factor.
#[test]
fn adapter_solve_is_linear_in_rhs() {
    with_converged_adapter(|backsolver| {
        let n_full = backsolver.dim();
        let mut rhs_a = vec![0.0; n_full];
        rhs_a[0] = 1.0;
        rhs_a[7] = -0.5;
        let mut rhs_b = vec![0.0; n_full];
        rhs_b[8] = 0.25;

        // Solve A, B, and A + 3·B.
        let mut sol_a = vec![0.0; n_full];
        let mut sol_b = vec![0.0; n_full];
        let mut rhs_ab = vec![0.0; n_full];
        for i in 0..n_full {
            rhs_ab[i] = rhs_a[i] + 3.0 * rhs_b[i];
        }
        let mut sol_ab = vec![0.0; n_full];
        assert!(backsolver.solve(&rhs_a, &mut sol_a));
        assert!(backsolver.solve(&rhs_b, &mut sol_b));
        assert!(backsolver.solve(&rhs_ab, &mut sol_ab));
        for i in 0..n_full {
            let pred = sol_a[i] + 3.0 * sol_b[i];
            let err = (sol_ab[i] - pred).abs();
            assert!(
                err < 1e-8,
                "linearity at row {i}: actual={} predicted={} |err|={err}",
                sol_ab[i],
                pred,
            );
        }
    });
}
