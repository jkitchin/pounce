//! End-to-end integration test mirroring upstream sIPOPT's
//! `examples/parametric_cpp` driver
//! ([`ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp`](../../../ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp)).
//!
//! Two complementary verifications run here:
//!
//! 1. **Upstream golden** (`parametric_cpp_matches_upstream_sipopt`):
//!    runs pounce's sensitivity step with the same perturbation
//!    `Δeta = (-0.5, 0)` that upstream sIPOPT uses in
//!    `parametric_driver.cpp`, then asserts per-component `Δx*` matches
//!    upstream's no-bound-check `printf` output to within 1e-8. The
//!    upstream numbers are baked into [`UPSTREAM_X_PERTURBED_NOBC`]
//!    after a one-shot run of `parametric_driver` built against the
//!    homebrew `sipopt-3.14.19` binary on 2026-05-14.
//!
//! 2. **Finite-difference cross-check** (`…_matches_finite_difference`):
//!    runs the sensitivity step with a small `Δeta1 = +0.01` and
//!    compares to a fresh resolve at the perturbed parameter. Validates
//!    pounce's sensitivity step is also internally consistent with
//!    pounce's own NLP solves, independently of upstream's exact
//!    floating-point trajectory.
//!
//! Reference for the math: Pirnay, López-Negrete & Biegler 2012, §3
//! (DOI: [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2),
//! verified via Crossref 2026-05-13).

use std::cell::RefCell;
use std::rc::Rc;

/// `bound_relax_factor`'s default, i.e. how far below its declared
/// lower bound of 0 the solver actually holds `x[2]`.
const BOUND_RELAX: f64 = 1e-8;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::boundcheck::RefineStop;
use pounce_sensitivity::{
    IndexSchurData, PdSensBacksolver, SensApplication, SensBacksolver, SensOptions,
};

/// Rust port of `ParametricTNLP` from upstream's `parametric_cpp`
/// example. n=5, m=4, FORTRAN-style triplets upstream are mapped to
/// C-style (0-based) in pounce.
struct ParametricTNLP {
    nominal_eta1: Number,
    nominal_eta2: Number,
}

impl ParametricTNLP {
    fn new(eta1: Number, eta2: Number) -> Self {
        Self {
            nominal_eta1: eta1,
            nominal_eta2: eta2,
        }
    }
}

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

        // Equalities g[0] = g[1] = 0; g[2] = nominal_eta1; g[3] = nominal_eta2.
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = self.nominal_eta1;
        b.g_u[2] = self.nominal_eta1;
        b.g_l[3] = self.nominal_eta2;
        b.g_u[3] = self.nominal_eta2;
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
        let (x1, x2, x3, eta1, eta2) = (x[0], x[1], x[2], x[3], x[4]);
        g[0] = 6.0 * x1 + 3.0 * x2 + 2.0 * x3 - eta1;
        g[1] = eta2 * x1 + x2 - x3 - 1.0;
        g[2] = eta1;
        g[3] = eta2;
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
                // Mirror upstream's row/col layout, but in 0-based form.
                // dg0/dx{0..3}, dg1/dx{0,1,2,4}, dg2/dx3, dg3/dx4.
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
                values[4] = x[4]; // eta2
                values[5] = 1.0;
                values[6] = -1.0;
                values[7] = x[0]; // x1
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
        // Lower-triangle entries: (0,0), (1,1), (2,2), (4,0) [off-diag,
        // x1 ↔ eta2 via the bilinear `eta2*x1` in g1]. Upstream uses
        // FORTRAN indexing and lists the off-diagonal twice (rows 1,5
        // and 5,1); pounce's triplet builder dedupes by lower-triangle
        // convention, so we list (4,0) once.
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

/// Solve at given `eta` and return `x*`.
fn solve_at(eta1: Number, eta2: Number) -> [Number; 5] {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(eta1, eta2)));

    // Capture x* via finalize_solution side-channel: re-wrap with a
    // capturing TNLP. Simpler: read off the final iterate from a
    // recording wrapper. For this test we use the on_converged
    // callback's data handle to read curr.x.
    let captured: Rc<RefCell<Option<[Number; 5]>>> = Rc::new(RefCell::new(None));
    let cap_for_cb = Rc::clone(&captured);
    app.set_on_converged(Box::new(move |data, _cq, _nlp, _pd| {
        let curr = data.borrow().curr.clone().expect("curr at convergence");
        let dx = curr
            .x
            .as_any()
            .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            .expect("x is dense");
        let v = dx.expanded_values();
        let mut out = [0.0; 5];
        out.copy_from_slice(&v[..5]);
        *cap_for_cb.borrow_mut() = Some(out);
    }));

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve_at({eta1}, {eta2}) failed: {status:?}",
    );
    let out = captured.borrow().expect("on_converged fired");
    out
}

/// Captured `printf` output of upstream sIPOPT's `parametric_driver`
/// (built against homebrew `ipopt 3.14.19` / `libsipopt.3.dylib`, run
/// on macOS 25.3.0 on 2026-05-14). The "without bound checking" block
/// — the std sensitivity step that pounce's `PdSensBacksolver` +
/// `SensApplication::parametric_step` mirror — reports these perturbed
/// x values at `Δeta1 = -0.5, Δeta2 = 0` from a nominal eta = (5, 1).
const UPSTREAM_X_PERTURBED_NOBC: [Number; 5] = [
    0.576_530_601_168_321_9,
    0.377_551_038_130_684_8,
    -0.045_918_360_700_993_31,
    4.500_000_000_000_000,
    1.000_000_000_000_000,
];

/// Same upstream run, nominal x* before perturbation. Used as the
/// linearization point so the test compares `Δx` rather than absolute
/// `x_perturbed` (which differs at the IPM-tolerance floor of ~1e-9
/// between upstream and pounce solves).
const UPSTREAM_X_NOMINAL: [Number; 5] = [
    0.632_653_057_519_998_2,
    0.387_755_107_968_002_7,
    0.020_408_165_488_001_08,
    5.000_000_000_000_000,
    1.000_000_000_000_000,
];

/// Run one parametric sensitivity step on a fresh nominal-eta solve
/// and return the primal Δx slice (first n_x entries of dx_full).
/// `delta_p` is the (Δeta1, Δeta2) perturbation.
fn run_sensitivity_step(delta_p: [Number; 2]) -> [Number; 5] {
    // Through the public method rather than a hand-built
    // `SensApplication`, so this carries the same barrier correction the
    // bounded step does and the two remain comparable.
    use pounce_sensitivity::Solver;

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "nominal solve failed: {status:?}",
    );
    let dx = solver
        .parametric_step(&[2, 3], &delta_p)
        .expect("parametric_step");
    std::array::from_fn(|i| dx[i])
}

#[test]
fn parametric_cpp_matches_upstream_sipopt() {
    // Upstream `parametric_driver.cpp` perturbs (eta1, eta2) from
    // (5, 1) to (4.5, 1) — see `parametricTNLP.cpp:18-19`.
    let delta_p = [-0.5, 0.0];
    let dx_sens = run_sensitivity_step(delta_p);

    // Upstream's "without bound checking" Δx is (perturbed - nominal)
    // — the linear sensitivity prediction, even though x[2] crosses
    // zero (active-set ignored, as required by the std step).
    let upstream_dx: [Number; 5] =
        std::array::from_fn(|i| UPSTREAM_X_PERTURBED_NOBC[i] - UPSTREAM_X_NOMINAL[i]);

    eprintln!(
        "Upstream-golden comparison @ Δeta = (-0.5, 0):\n  upstream Δx = {:?}\n  pounce Δx   = {:?}",
        upstream_dx, dx_sens,
    );

    // Acceptance: per-component agreement to 1e-8, matching the issue's
    // "matches upstream to 1e-8" criterion.
    for k in 0..5 {
        let err = (dx_sens[k] - upstream_dx[k]).abs();
        assert!(
            err < 1e-8,
            "dx[{k}]: pounce={}, upstream={}, |err|={err} not < 1e-8",
            dx_sens[k],
            upstream_dx[k],
        );
    }
}

#[test]
fn parametric_cpp_first_order_sensitivity_matches_finite_difference() {
    // Step size chosen well above tol=1e-8 (so the solver's residual
    // floor doesn't dominate the FD numerator) and well below the
    // active-set boundary (x[2] crosses zero somewhere between
    // eta1 ≈ 4.9 and 4.5, so 0.01 stays in the interior).
    let dx_step: Number = 1.0e-2;
    let eta1_nominal: Number = 5.0;
    let eta2_nominal: Number = 1.0;

    // 1) Solve at nominal twice: once to capture x*, once with the
    //    sensitivity step wired in. Two separate solves so the
    //    `captured` state in solve_at doesn't conflict with the
    //    sensitivity-step closure below.
    let x_nominal = solve_at(eta1_nominal, eta2_nominal);

    // 2) Forward sensitivity step at nominal: register a callback that
    //    runs the PdSensBacksolver-driven SensApplication pipeline and
    //    stashes Δx into a side channel.
    let dx_full_out: Rc<RefCell<Option<Vec<Number>>>> = Rc::new(RefCell::new(None));
    let dx_full_clone = Rc::clone(&dx_full_out);

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(
        eta1_nominal,
        eta2_nominal,
    )));

    app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
        // Compound-vector block offsets at convergence. For
        // ParametricTNLP: n_x=5, n_s=0, n_c=4, n_d=0 → y_c block
        // starts at flat offset 5. Constraints 2 and 3 (the parameter
        // pins g[2]=eta1, g[3]=eta2) live at flat indices 7 and 8.
        let curr = data.borrow().curr.clone().expect("curr at convergence");
        let n_x = curr.x.dim() as usize;
        let n_s = curr.s.dim() as usize;
        let y_c_offset = n_x + n_s;
        let param_rows = vec![
            (y_c_offset + 2) as pounce_common::types::Index,
            (y_c_offset + 3) as pounce_common::types::Index,
        ];

        let backsolver =
            PdSensBacksolver::new(data, cq, nlp, pd).expect("PdSensBacksolver construction");
        let n_full = backsolver.dim();

        // `SensApplication::parametric_step` matches upstream
        // [`SensStdStepCalc::Step`](../../../ref/Ipopt/contrib/sIPOPT/src/SensStdStepCalc.cpp:48-83)
        // — scatter Δp onto the y_c slots picked by `a_data`, then one
        // backsolve against the converged KKT factor.
        let a_data = IndexSchurData::from_parts(param_rows, vec![1, 1]).expect("A SchurData");
        let opts = SensOptions {
            run_sens: true,
            ..SensOptions::default()
        };
        let sens_app = SensApplication::new(a_data, backsolver, opts);

        let delta_p: Vec<Number> = vec![dx_step, 0.0]; // (Δeta1, Δeta2).
        let mut dx_full = vec![0.0; n_full];
        assert!(
            sens_app.parametric_step(&delta_p, &mut dx_full),
            "SensApplication::parametric_step failed"
        );
        *dx_full_clone.borrow_mut() = Some(dx_full);
    }));

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "nominal solve for sensitivity step failed: {status:?}",
    );

    let dx_full = dx_full_out
        .borrow()
        .clone()
        .expect("on_converged populated dx_full");

    // 3) Finite-difference reference: resolve at perturbed eta.
    let eta1_perturbed = eta1_nominal + dx_step;
    let x_perturbed = solve_at(eta1_perturbed, eta2_nominal);
    let dx_fd: [Number; 5] = std::array::from_fn(|i| x_perturbed[i] - x_nominal[i]);

    // The sensitivity step's primal Δx lives in dx_full[0..5]. Drop
    // sign-convention assumptions and absolute-value compare to FD —
    // alternatively, infer the convention from the parameter pin
    // (Δx[3] should track Δeta1 exactly).
    let dx_x: [Number; 5] = std::array::from_fn(|i| dx_full[i]);

    eprintln!(
        "ParametricTNLP sensitivity test:\n  x_nom = {:?}\n  x_pert = {:?}\n  dx_fd  = {:?}\n  dx_sens= {:?}",
        x_nominal, x_perturbed, dx_fd, dx_x
    );

    // Architectural acceptance: dx_full[3] (the eta1 slot) should
    // be roughly ±dx_step — the parameter is pinned by the constraint
    // we perturbed.
    assert!(
        (dx_x[3].abs() - dx_step).abs() < 1e-6,
        "Δeta1 slot (dx_full[3]) magnitude {} differs from |Δp|={} by more than 1e-6",
        dx_x[3].abs(),
        dx_step,
    );

    // Δeta2 slot should be ~zero (no perturbation on eta2).
    assert!(
        dx_x[4].abs() < 1e-7,
        "Δeta2 slot (dx_full[4]) = {} not near zero",
        dx_x[4],
    );

    // The non-parameter primals should match FD up to first-order
    // accuracy. Determine the sign by inspecting dx_x[3] vs dx_fd[3]:
    // both are pinned to the parameter, so they must have the same
    // sign convention.
    let sign = (dx_x[3] * dx_fd[3]).signum();
    assert!(
        sign > 0.0,
        "sensitivity sign convention disagrees with FD reference: dx_x[3]={}, dx_fd[3]={}",
        dx_x[3],
        dx_fd[3]
    );

    for k in 0..3 {
        let pred = sign * dx_x[k];
        let err = (pred - dx_fd[k]).abs();
        assert!(
            err < 1e-6,
            "dx[{k}]: sens (signed)={pred}, fd={}, |err|={err} not < 1e-6",
            dx_fd[k],
        );
    }
}

// ── fix-relax: pin the crossing variable and re-solve ────────────────────
//
// At the perturbation upstream's own driver uses, `Delta eta = (-0.5, 0)`,
// the plain step drives `x[2]` to -0.046 against its lower bound of 0. The
// single-pass clamp truncates `x[2]` there and leaves every other
// coordinate at its linear-predictor value. `parametric_step_bounded`
// instead pins `x[2]` at the bound and re-solves, so the others move to
// stay consistent under the pin.

/// Solve at the nominal parameters and run `parametric_step_bounded`.
fn run_bounded_step(delta_p: [Number; 2]) -> ([Number; 5], Vec<Index>) {
    let (dx, pinned, _) = run_bounded_step_with_stop(delta_p, 8);
    (dx, pinned)
}

/// `run_bounded_step` with the pass limit and the stop reason exposed.
fn run_bounded_step_with_stop(
    delta_p: [Number; 2],
    max_iter: usize,
) -> ([Number; 5], Vec<Index>, RefineStop) {
    use pounce_sensitivity::Solver;

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "nominal solve failed: {status:?}",
    );

    let (dx, pinned, stop) = solver
        .parametric_step_bounded(&[2, 3], &delta_p, max_iter, None)
        .expect("parametric_step_bounded");
    (std::array::from_fn(|i| dx[i]), pinned, stop)
}

#[test]
fn fix_relax_pins_the_crossing_variable_at_its_bound() {
    let base = solve_at(5.0, 1.0);
    let plain = run_sensitivity_step([-0.5, 0.0]);
    let (fixed, pinned) = run_bounded_step([-0.5, 0.0]);

    assert_eq!(pinned, vec![2], "x[2] is the coordinate that crosses here");

    // the plain step leaves the bound, which is why upstream's own
    // no-bound-check output reports a negative x[2]
    let plain_x2 = base[2] + plain[2];
    assert!(
        plain_x2 < -1e-3,
        "expected the plain step to violate x[2] >= 0, got {plain_x2}",
    );

    // and the pinned re-solve puts it back on the bound
    let fixed_x2 = base[2] + fixed[2];
    // "On its bound" means the bound the solver actually holds, which
    // `bound_relax_factor` (1e-8) has moved from 0 to -1e-8. A pinned
    // re-solve that is accurate enough lands on -1e-8 exactly, so the
    // window has to include it rather than stop one ulp short of it.
    assert!(
        fixed_x2.abs() <= BOUND_RELAX + 1e-12,
        "fix-relax should land x[2] on its bound, got {fixed_x2}",
    );
}

#[test]
fn fix_relax_tracks_the_resolve_more_closely_than_the_clamp() {
    let base = solve_at(5.0, 1.0);
    let exact = solve_at(4.5, 1.0);
    let plain = run_sensitivity_step([-0.5, 0.0]);
    let (fixed, _) = run_bounded_step([-0.5, 0.0]);

    // the clamp: truncate the crossing coordinate, leave the rest at
    // their linear-predictor values
    let clamped: [Number; 5] = std::array::from_fn(|k| {
        let v = base[k] + plain[k];
        if k < 3 { v.max(0.0) } else { v }
    });
    let fixed_x: [Number; 5] = std::array::from_fn(|k| base[k] + fixed[k]);

    let err = |v: &[Number; 5]| -> Number {
        (0..5)
            .map(|k| (v[k] - exact[k]).abs())
            .fold(0.0, Number::max)
    };
    let (e_clamp, e_fixed) = (err(&clamped), err(&fixed_x));
    println!("exact   {exact:?}");
    println!("clamped {clamped:?}  max err {e_clamp:.3e}");
    println!("fixed   {fixed_x:?}  max err {e_fixed:.3e}");

    assert!(
        e_fixed < e_clamp * 1e-3,
        "fix-relax ({e_fixed:.3e}) should track the re-solve far better \
         than the clamp ({e_clamp:.3e})",
    );
}

#[test]
fn a_step_that_stays_inside_the_bounds_pins_nothing() {
    let (bounded, pinned) = run_bounded_step([0.01, 0.0]);
    let plain = run_sensitivity_step([0.01, 0.0]);
    assert!(pinned.is_empty(), "nothing crosses at this perturbation");
    for k in 0..5 {
        assert!(
            (bounded[k] - plain[k]).abs() < 1e-14,
            "with no crossing the bounded step must equal the plain one",
        );
    }
}

#[test]
fn a_pin_beyond_the_degrees_of_freedom_is_refused() {
    // ParametricTNLP has five variables and four constraints, so one
    // degree of freedom. Pinning x[2] consumes it. The step at this
    // perturbation also drives x[0] out, but no step can hold both at
    // their bounds and satisfy the constraints, so the augmented system
    // is singular and the second pin is refused. The refinement keeps
    // what it could achieve rather than returning the singular solve.
    let base = solve_at(5.0, 1.0);
    let (fixed, pinned, stop) = run_bounded_step_with_stop([-2.0, -0.5], 8);
    assert_eq!(pinned, vec![2], "only the first pin fits in one DOF");
    assert_eq!(
        stop,
        RefineStop::DegreesOfFreedom,
        "and the caller is told which limit that was, not that a budget          ran out",
    );

    let x2 = base[2] + fixed[2];
    assert!(x2.abs() < 1e-7, "the pin that was accepted holds: {x2}");
    for k in 0..5 {
        assert!(
            (base[k] + fixed[k]).abs() < 1e3,
            "a refused pin must not leak a singular solve into the step",
        );
    }
}

// ── a model with room for several pins ───────────────────────────────────
//
// `ParametricTNLP` has one degree of freedom, so it can only ever accept
// one pin. This one has three: four variables against a single pin
// constraint. Its unconstrained minimum is `x = (p, 2p, 3p)` and all
// three carry a lower bound of 0, so a perturbation to a negative `p`
// drives every one of them out at once and the exact answer is the
// origin.

struct ThreeFreeTNLP {
    nominal_p: Number,
}

impl TNLP for ThreeFreeTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 7,
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
        b.g_l[0] = self.nominal_p;
        b.g_u[0] = self.nominal_p;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 1.0;
        sp.x[1] = 1.0;
        sp.x[2] = 1.0;
        sp.x[3] = self.nominal_p;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let p = x[3];
        Some((x[0] - p).powi(2) + (x[1] - 2.0 * p).powi(2) + (x[2] - 3.0 * p).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let p = x[3];
        g[0] = 2.0 * (x[0] - p);
        g[1] = 2.0 * (x[1] - 2.0 * p);
        g[2] = 2.0 * (x[2] - 3.0 * p);
        g[3] = -2.0 * (x[0] - p) - 4.0 * (x[1] - 2.0 * p) - 6.0 * (x[2] - 3.0 * p);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[3];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index]);
                jcol.copy_from_slice(&[3 as Index]);
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // lower triangle of the objective Hessian; g is linear
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2, 3, 3, 3, 3]);
                jcol.copy_from_slice(&[0, 1, 2, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                values[2] = 2.0 * obj_factor;
                values[3] = -2.0 * obj_factor;
                values[4] = -4.0 * obj_factor;
                values[5] = -6.0 * obj_factor;
                values[6] = 28.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn three_free_solve_at(p: Number) -> [Number; 4] {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ThreeFreeTNLP { nominal_p: p }));
    let captured: Rc<RefCell<Option<[Number; 4]>>> = Rc::new(RefCell::new(None));
    let cap = Rc::clone(&captured);
    app.set_on_converged(Box::new(move |data, _cq, _nlp, _pd| {
        let curr = data.borrow().curr.clone().expect("curr");
        let v = curr
            .x
            .as_any()
            .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            .expect("dense")
            .expanded_values();
        *cap.borrow_mut() = Some(std::array::from_fn(|i| v[i]));
    }));
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "three_free_solve_at({p}) failed: {status:?}",
    );
    captured.borrow().expect("on_converged fired")
}

/// The three-free model, solved at its nominal parameter.
fn three_free_solver() -> pounce_sensitivity::Solver {
    use pounce_sensitivity::Solver;

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ThreeFreeTNLP { nominal_p: 1.0 }));
    let mut solver = Solver::new(app, tnlp);
    assert!(matches!(
        solver.solve(),
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    solver
}

#[test]
fn fix_relax_pins_three_crossings_at_once() {
    let solver = three_free_solver();

    let base = three_free_solve_at(1.0);
    let exact = three_free_solve_at(-1.0);
    let plain = solver.parametric_step(&[0], &[-2.0]).expect("plain step");
    let (fixed, pinned, stop) = solver
        .parametric_step_bounded(&[0], &[-2.0], 8, None)
        .expect("bounded step");
    assert_eq!(
        stop,
        RefineStop::Settled,
        "the loop ends on an empty violation list, not on the pass limit",
    );

    // dx/dp is (1, 2, 3), so the step drives all three below zero and
    // the worst violator is taken first
    assert_eq!(pinned, vec![2, 1, 0], "worst first, then the next two");

    // three pins fit inside three degrees of freedom, so every one holds
    for k in 0..3 {
        let v = base[k] + fixed[k];
        assert!(v.abs() < 1e-7, "x[{k}] should sit on its bound, got {v}");
        assert!((exact[k]).abs() < 1e-6, "the exact answer is the origin");
    }

    let err = |v: &[Number; 4]| -> Number {
        (0..4)
            .map(|k| (v[k] - exact[k]).abs())
            .fold(0.0, Number::max)
    };
    let clamped: [Number; 4] = std::array::from_fn(|k| {
        let v = base[k] + plain[k];
        if k < 3 { v.max(0.0) } else { v }
    });
    let fixed_x: [Number; 4] = std::array::from_fn(|k| base[k] + fixed[k]);
    println!(
        "exact {exact:?}\nclamp {clamped:?} err {:.3e}\nfixed {fixed_x:?} err {:.3e}",
        err(&clamped),
        err(&fixed_x)
    );
    assert!(err(&fixed_x) <= err(&clamped));
}

#[test]
fn the_pass_limit_no_longer_picks_the_answer() {
    // gh#732. A pass used to take the single worst crossing, so the
    // number of crossings repaired was whatever `max_iter` allowed and
    // the answer moved with the budget. This model's step crosses three
    // bounds at once: one pass now takes all three, and every budget
    // from one pass up agrees to the last digit.
    let solver = three_free_solver();
    let at = |k| {
        solver
            .parametric_step_bounded(&[0], &[-2.0], k, None)
            .expect("bounded step")
    };
    let (one, pins_one, stop_one) = at(1);
    let (eight, pins_eight, stop_eight) = at(8);

    assert_eq!(stop_one, RefineStop::Settled, "one pass settles it");
    assert_eq!(stop_eight, RefineStop::Settled);
    assert_eq!(pins_one, vec![2, 1, 0], "all three, worst first");
    assert_eq!(pins_one, pins_eight, "and the budget does not change them");
    for k in 0..4 {
        assert!(
            (one[k] - eight[k]).abs() < 1e-14,
            "x[{k}] moved with the budget: {} vs {}",
            one[k],
            eight[k],
        );
    }
}

/// Walk the same perturbation the fix-relax tests use. Returns the
/// primal step and the breakpoints crossed.
fn run_path_step(
    delta_p: [Number; 2],
) -> (
    [Number; 5],
    Vec<pounce_sensitivity::boundcheck::PathSegment>,
) {
    use pounce_sensitivity::Solver;

    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "nominal solve failed: {status:?}",
    );
    let (dx, segs) = solver
        .parametric_step_path(&[2, 3], &delta_p, 8)
        .expect("parametric_step_path");
    (std::array::from_fn(|i| dx[i]), segs)
}

#[test]
fn path_agrees_with_the_plain_step_when_nothing_crosses() {
    // A perturbation small enough to keep every variable inside its
    // bounds leaves nothing for the walk to do, so it must reproduce
    // the plain step rather than merely come close.
    let plain = run_sensitivity_step([-0.01, 0.0]);
    let (walked, segs) = run_path_step([-0.01, 0.0]);
    assert!(segs.is_empty(), "no breakpoint expected, got {segs:?}");
    for i in 0..5 {
        assert!(
            (walked[i] - plain[i]).abs() < 1e-12,
            "coordinate {i}: walk {} vs plain {}",
            walked[i],
            plain[i],
        );
    }
}

#[test]
fn path_stops_where_the_variable_reaches_its_bound() {
    let base = solve_at(5.0, 1.0);
    let (walked, segs) = run_path_step([-0.5, 0.0]);

    assert_eq!(segs.len(), 1, "one crossing expected, got {segs:?}");
    assert_eq!(segs[0].var_row, 2, "x[2] is what crosses here");
    assert!(segs[0].lower, "x[2] reaches its lower bound here");
    assert!(segs[0].pinned, "it reaches a bound, so it pins");
    // The plain step puts x[2] at -0.0459 for the whole perturbation,
    // and x[2] starts at 0.0204, so the bound is reached partway.
    assert!(
        segs[0].at > 0.0 && segs[0].at < 1.0,
        "the crossing is interior to the perturbation, got {}",
        segs[0].at,
    );
    // Having pinned there, the walk must leave it on the bound.
    let x2 = base[2] + walked[2];
    // See `fix_relax_pins_the_crossing_variable_at_its_bound`: the held
    // bound is the relaxed one, at -`BOUND_RELAX`.
    assert!(
        x2.abs() <= BOUND_RELAX + 1e-12,
        "x[2] should finish on its bound, got {x2}",
    );
}

#[test]
fn path_and_fix_relax_agree_on_a_single_crossing() {
    // Both repair the same one condition on this model, so they must
    // land on the same point. They separate only when crossings
    // interact, which needs more than one.
    let (walked, _) = run_path_step([-0.5, 0.0]);
    let (fixed, _) = run_bounded_step([-0.5, 0.0]);
    for i in 0..5 {
        assert!(
            (walked[i] - fixed[i]).abs() < 1e-7,
            "coordinate {i}: path {} vs fix_relax {}",
            walked[i],
            fixed[i],
        );
    }
}
