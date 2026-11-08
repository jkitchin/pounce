//! The matrix-form QP / conic entry (gh#1147 S3-I2): `PounceQp*` handle,
//! builders, `PounceQpSolve`, the getters and the report writer, driven
//! through the exported C functions.
//!
//! Each model is tiny and has an optimum known by hand. The full per-cone and
//! per-error matrix, and parity against Python's `solve_qp` / `solve_socp`, are
//! S3-I3's; what is here is one model per path (LP, convex QP, SOC, exp) on
//! each engine, every getter's NULL / before-solve / after-a-refused-solve
//! branch, and the report.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_cinterface::*;
use std::ffi::CString;
use std::ptr::{null, null_mut};

const TOL: f64 = 1e-6;

fn cs(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Owns a handle so a failing assertion still frees it.
struct Qp(PounceQp);
impl Drop for Qp {
    fn drop(&mut self) {
        unsafe { PounceFreeQp(self.0) }
    }
}

fn create(n: Index, m_eq: Index, m_ineq: Index, style: Index) -> Qp {
    let p = unsafe { PounceCreateQp(n, m_eq, m_ineq, style) };
    assert!(!p.is_null());
    Qp(p)
}

fn objective(qp: &Qp, p: &[(Index, Index, f64)], c: &[f64]) -> Bool {
    let (r, cc, v) = split(p);
    unsafe {
        PounceQpSetObjective(
            qp.0,
            r.len() as Index,
            r.as_ptr(),
            cc.as_ptr(),
            v.as_ptr(),
            c.as_ptr(),
        )
    }
}

fn equalities(qp: &Qp, a: &[(Index, Index, f64)], b: &[f64]) -> Bool {
    let (r, c, v) = split(a);
    unsafe {
        PounceQpSetEqualities(
            qp.0,
            r.len() as Index,
            r.as_ptr(),
            c.as_ptr(),
            v.as_ptr(),
            b.as_ptr(),
        )
    }
}

fn inequalities(qp: &Qp, g: &[(Index, Index, f64)], h: &[f64]) -> Bool {
    let (r, c, v) = split(g);
    unsafe {
        PounceQpSetInequalities(
            qp.0,
            r.len() as Index,
            r.as_ptr(),
            c.as_ptr(),
            v.as_ptr(),
            h.as_ptr(),
        )
    }
}

fn split(t: &[(Index, Index, f64)]) -> (Vec<Index>, Vec<Index>, Vec<f64>) {
    (
        t.iter().map(|e| e.0).collect(),
        t.iter().map(|e| e.1).collect(),
        t.iter().map(|e| e.2).collect(),
    )
}

fn str_opt(qp: &Qp, k: &str, v: &str) -> Bool {
    unsafe { PounceQpSetStrOption(qp.0, cs(k).as_ptr(), cs(v).as_ptr()) }
}

fn num_opt(qp: &Qp, k: &str, v: f64) -> Bool {
    unsafe { PounceQpSetNumOption(qp.0, cs(k).as_ptr(), v) }
}

struct Sol {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    z_lb: Vec<f64>,
    z_ub: Vec<f64>,
    obj: f64,
    iters: Index,
}

fn solution(qp: &Qp, n: usize, m_eq: usize, m_ineq: usize) -> Sol {
    let mut s = Sol {
        x: vec![f64::NAN; n],
        y: vec![f64::NAN; m_eq],
        z: vec![f64::NAN; m_ineq],
        z_lb: vec![f64::NAN; n],
        z_ub: vec![f64::NAN; n],
        obj: f64::NAN,
        iters: -1,
    };
    let ok = unsafe {
        PounceQpGetSolution(
            qp.0,
            s.x.as_mut_ptr(),
            s.y.as_mut_ptr(),
            s.z.as_mut_ptr(),
            s.z_lb.as_mut_ptr(),
            s.z_ub.as_mut_ptr(),
            &mut s.obj,
            &mut s.iters,
        )
    };
    assert_eq!(ok, 1);
    s
}

fn residuals(qp: &Qp) -> PounceQpResiduals {
    let mut r = PounceQpResiduals::default();
    assert_eq!(unsafe { PounceQpGetResiduals(qp.0, &mut r) }, 1);
    r
}

fn certificate(qp: &Qp) -> PounceQpCertificate {
    let mut c = PounceQpCertificate::default();
    assert_eq!(unsafe { PounceQpGetCertificate(qp.0, &mut c) }, 1);
    c
}

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < TOL, "{a:?} vs {b:?}");
    }
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// LP: `min −x0 − 2 x1` s.t. `x0 − x1 = 0`, `x0 + x1 ≤ 1`, `x ≥ 0`.
/// Optimum `x = (½, ½)`, `obj = −3/2`; stationarity `c + Aᵀy + Gᵀz = 0`
/// gives `z = 3/2`, `y = −½`.
fn lp() -> Qp {
    let qp = create(2, 1, 1, 0);
    assert_eq!(objective(&qp, &[], &[-1.0, -2.0]), 1);
    assert_eq!(equalities(&qp, &[(0, 0, 1.0), (0, 1, -1.0)], &[0.0]), 1);
    assert_eq!(inequalities(&qp, &[(0, 0, 1.0), (0, 1, 1.0)], &[1.0]), 1);
    assert_eq!(
        unsafe { PounceQpSetBounds(qp.0, [0.0, 0.0].as_ptr(), null()) },
        1
    );
    qp
}

fn check_lp(qp: &Qp) {
    let s = solution(qp, 2, 1, 1);
    close(&s.x, &[0.5, 0.5]);
    close(&[s.obj], &[-1.5]);
    close(&s.y, &[-0.5]);
    close(&s.z, &[1.5]);
    close(&s.z_lb, &[0.0, 0.0]);
    close(&s.z_ub, &[0.0, 0.0]);
    // Presolve can settle this model alone (0 iterations).
    assert!(s.iters >= 0);
    let r = residuals(qp);
    assert!(r.kkt_error <= 1e-7 && r.kkt_error_raw <= 1e-6, "{r:?}");
}

/// Convex QP, 1-based triplets: `min ½(x0² + x1²) − x0 − x1` s.t.
/// `x0 + x1 ≤ 1`, `0 ≤ x ≤ 10`. Optimum `x = (½, ½)`, `obj = −3/4`, `z = ½`.
fn convex_qp() -> Qp {
    let qp = create(2, 0, 1, 1);
    assert_eq!(
        objective(&qp, &[(1, 1, 1.0), (2, 2, 1.0)], &[-1.0, -1.0]),
        1
    );
    assert_eq!(inequalities(&qp, &[(1, 1, 1.0), (1, 2, 1.0)], &[1.0]), 1);
    let (lb, ub) = ([0.0, 0.0], [10.0, 10.0]);
    assert_eq!(
        unsafe { PounceQpSetBounds(qp.0, lb.as_ptr(), ub.as_ptr()) },
        1
    );
    qp
}

fn check_convex_qp(qp: &Qp) {
    let s = solution(qp, 2, 0, 1);
    close(&s.x, &[0.5, 0.5]);
    close(&[s.obj], &[-0.75]);
    close(&s.z, &[0.5]);
}

/// SOC: `min t` s.t. `(t, x1 − 2, x2 + 1) ∈ Q³` and `x1 + x2 = 0` — the
/// distance from `(2, −1)` to the line `x1 + x2 = 0`: `t = 1/√2` at
/// `x = (3/2, −3/2)`. Variables `(t, x1, x2)`; `G = −I`, `h = (0, −2, 1)`.
fn soc() -> Qp {
    let qp = create(3, 1, 3, 0);
    assert_eq!(objective(&qp, &[], &[1.0, 0.0, 0.0]), 1);
    assert_eq!(equalities(&qp, &[(0, 1, 1.0), (0, 2, 1.0)], &[0.0]), 1);
    assert_eq!(
        inequalities(
            &qp,
            &[(0, 0, -1.0), (1, 1, -1.0), (2, 2, -1.0)],
            &[0.0, -2.0, 1.0]
        ),
        1
    );
    assert_eq!(unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_SOC, 3, 0.0) }, 0);
    qp
}

// ---------------------------------------------------------------------------
// Solves
// ---------------------------------------------------------------------------

#[test]
fn lp_on_the_ipm() {
    let qp = lp();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    check_lp(&qp);
    assert!(solution(&qp, 2, 1, 1).iters > 0);
    let c = certificate(&qp);
    assert_eq!(c.kind, POUNCE_QP_CERT_NONE);
    assert!(c.scale.is_nan());
    // The default driver is the homogeneous embedding: τ > 0, κ ≈ 0.
    assert!(c.tau > 0.0 && c.kappa.is_finite(), "{c:?}");
}

#[test]
fn lp_presolved_with_crossover_and_on_the_direct_driver() {
    let qp = lp();
    assert_eq!(str_opt(&qp, "qp_presolve", "yes"), 1);
    assert_eq!(str_opt(&qp, "qp_crossover", "yes"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    check_lp(&qp);

    let qp = lp();
    assert_eq!(str_opt(&qp, "qp_hsde", "no"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    check_lp(&qp);
    // No embedding on the direct driver.
    let c = certificate(&qp);
    assert!(c.tau.is_nan() && c.kappa.is_nan(), "{c:?}");
}

#[test]
fn lp_on_the_active_set_engine() {
    let qp = lp();
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    check_lp(&qp);
}

#[test]
fn convex_qp_on_both_engines_with_one_based_triplets() {
    let qp = convex_qp();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    check_convex_qp(&qp);

    let qp = convex_qp();
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    assert_eq!(str_opt(&qp, "qp_presolve", "yes"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    check_convex_qp(&qp);
}

#[test]
fn soc_distance_to_a_line() {
    let qp = soc();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    let s = solution(&qp, 3, 1, 3);
    let r = std::f64::consts::FRAC_1_SQRT_2;
    close(&s.x, &[r, 1.5, -1.5]);
    close(&[s.obj], &[r]);
    let res = residuals(&qp);
    // Conic: both sets are the plain residuals.
    assert_eq!(res.kkt_error, res.kkt_error_raw);
    assert!(res.kkt_error < 1e-6, "{res:?}");
}

/// The geometric program in `solve_socp`'s docstring (`python/pounce/qp.py`):
/// `min x + 1/x = min_u eᵘ + e⁻ᵘ`, optimum 2 at `u = 0`.
#[test]
fn exponential_cone_geometric_program() {
    let qp = create(3, 0, 6, 0);
    assert_eq!(objective(&qp, &[], &[0.0, 1.0, 1.0]), 1);
    let g = [(0, 0, -1.0), (2, 1, -1.0), (3, 0, 1.0), (5, 2, -1.0)];
    assert_eq!(inequalities(&qp, &g, &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]), 1);
    for k in 0..2 {
        assert_eq!(unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_EXP, 3, 0.0) }, k);
    }
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    let s = solution(&qp, 3, 0, 6);
    assert!((s.obj - 2.0).abs() < 1e-5, "obj = {}", s.obj);
    assert!(s.x[0].abs() < 1e-3, "u = {}", s.x[0]);
}

#[test]
fn infeasible_and_unbounded_certificates() {
    // −x0 ≤ −1 with x0 ≤ 0: primal infeasible.
    let qp = create(1, 0, 1, 0);
    assert_eq!(objective(&qp, &[], &[1.0]), 1);
    assert_eq!(inequalities(&qp, &[(0, 0, -1.0)], &[-1.0]), 1);
    assert_eq!(
        unsafe { PounceQpSetBounds(qp.0, null(), [0.0].as_ptr()) },
        1
    );
    assert_eq!(
        unsafe { PounceQpSolve(qp.0) },
        POUNCE_QP_STATUS_PRIMAL_INFEASIBLE
    );
    let c = certificate(&qp);
    assert_eq!(c.kind, POUNCE_QP_CERT_PRIMAL_INFEASIBLE);
    assert!(c.scale > 0.0 && c.scale.is_finite(), "{c:?}");

    // min −x0 over x0 ≥ 0: unbounded.
    let qp = create(1, 0, 0, 0);
    assert_eq!(objective(&qp, &[], &[-1.0]), 1);
    assert_eq!(
        unsafe { PounceQpSetBounds(qp.0, [0.0].as_ptr(), null()) },
        1
    );
    assert_eq!(
        unsafe { PounceQpSolve(qp.0) },
        POUNCE_QP_STATUS_DUAL_INFEASIBLE
    );
    let c = certificate(&qp);
    assert_eq!(c.kind, POUNCE_QP_CERT_DUAL_INFEASIBLE);
    let s = solution(&qp, 1, 0, 0);
    assert_eq!(c.scale, s.x[0].abs());
}

/// gh#112: an indefinite `P` is refused under the IPM (not certified PSD),
/// solved under the active-set engine, and a caller's claim replaces the
/// verdict. `min ½x0² − ½x1²` over `[−1, 1]²`: local minima `x = (0, ±1)`,
/// `obj = −½`.
#[test]
fn indefinite_p_is_refused_under_the_ipm_only() {
    let make = || {
        let qp = create(2, 0, 0, 0);
        assert_eq!(objective(&qp, &[(0, 0, 1.0), (1, 1, -1.0)], &[0.0, 0.0]), 1);
        let (lb, ub) = ([-1.0, -1.0], [1.0, 1.0]);
        assert_eq!(
            unsafe { PounceQpSetBounds(qp.0, lb.as_ptr(), ub.as_ptr()) },
            1
        );
        qp
    };
    let qp = make();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_INDEFINITE);
    let qp = make();
    assert_eq!(str_opt(&qp, "hessian_inertia", "indefinite"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_INDEFINITE);

    let qp = make();
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    let code = unsafe { PounceQpSolve(qp.0) };
    assert!(code >= 0, "active-set refused an indefinite P: {code}");
    let s = solution(&qp, 2, 0, 0);
    assert_eq!(code, POUNCE_QP_STATUS_OPTIMAL, "x = {:?}", s.x);
    assert!((s.obj + 0.5).abs() < TOL, "obj = {}", s.obj);

    // A PSD P under a coupled pattern is certified, not just a diagonal one.
    let qp = create(2, 0, 0, 0);
    let p = [(0, 0, 2.0), (1, 0, -1.0), (1, 1, 2.0)];
    assert_eq!(objective(&qp, &p, &[-1.0, -1.0]), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    close(&solution(&qp, 2, 0, 0).x, &[1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// error-path: PounceCreateQp
// error-path: PounceQpSolve
#[test]
fn solve_refusals_discard_the_previous_result() {
    assert!(unsafe { PounceCreateQp(0, 0, 0, 0) }.is_null());
    assert_eq!(unsafe { PounceQpSolve(null_mut()) }, POUNCE_QP_ERR_NULL);

    // No objective.
    let qp = create(1, 0, 0, 0);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_INCOMPLETE);
    // Declared rows never set.
    let qp = create(1, 1, 0, 0);
    assert_eq!(objective(&qp, &[], &[1.0]), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_INCOMPLETE);

    // Cones covering 2 of 3 rows; then PSD mixed with exp.
    let qp = soc();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    assert_eq!(
        unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_NONNEG, 2, 0.0) },
        1
    );
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_CONES);
    // ... and the earlier result is gone.
    let mut r = PounceQpResiduals::default();
    assert_eq!(unsafe { PounceQpGetResiduals(qp.0, &mut r) }, 0);

    let qp = create(1, 0, 4, 0);
    assert_eq!(objective(&qp, &[], &[1.0]), 1);
    assert_eq!(inequalities(&qp, &[], &[0.0; 4]), 1);
    assert_eq!(unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_PSD, 1, 0.0) }, 0);
    assert_eq!(unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_EXP, 3, 0.0) }, 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_CONES);

    // Options: an IPM-only switch on the active-set engine; a QP-only option
    // with cones.
    let qp = lp();
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    assert_eq!(str_opt(&qp, "qp_hsde", "yes"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_OPTIONS);
    let qp = soc();
    assert_eq!(num_opt(&qp, "tau", 0.9), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_OPTIONS);
    let qp = soc();
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_OPTIONS);
}

// error-path: PounceQpSetObjective
// error-path: PounceQpSetEqualities
// error-path: PounceQpSetInequalities
// error-path: PounceQpSetBounds
#[test]
fn builders_refuse_bad_input_and_change_nothing() {
    unsafe {
        let c = [1.0, 1.0];
        assert_eq!(
            PounceQpSetObjective(null_mut(), 0, null(), null(), null(), c.as_ptr()),
            0
        );
        assert_eq!(
            PounceQpSetEqualities(null_mut(), 0, null(), null(), null(), null()),
            0
        );
        assert_eq!(
            PounceQpSetInequalities(null_mut(), 0, null(), null(), null(), null()),
            0
        );
        assert_eq!(PounceQpSetBounds(null_mut(), null(), null()), 0);
    }
    let qp = create(2, 1, 1, 0);
    // c is required; NULL triplet arrays with nnz > 0; negative nnz.
    assert_eq!(
        unsafe { PounceQpSetObjective(qp.0, 0, null(), null(), null(), null()) },
        0
    );
    assert_eq!(
        unsafe { PounceQpSetObjective(qp.0, 1, null(), null(), null(), [0.0, 0.0].as_ptr()) },
        0
    );
    assert_eq!(
        unsafe { PounceQpSetObjective(qp.0, -1, null(), null(), null(), [0.0, 0.0].as_ptr()) },
        0
    );
    // Upper triangle, out of range, non-finite.
    assert_eq!(objective(&qp, &[(0, 1, 1.0)], &[0.0, 0.0]), 0);
    assert_eq!(objective(&qp, &[(2, 0, 1.0)], &[0.0, 0.0]), 0);
    assert_eq!(objective(&qp, &[(0, 0, f64::NAN)], &[0.0, 0.0]), 0);
    assert_eq!(objective(&qp, &[], &[0.0, f64::INFINITY]), 0);
    // A row index past m_eq / m_ineq, a column past n, a 1-based index on a
    // 0-based handle reaching past the end.
    assert_eq!(equalities(&qp, &[(1, 0, 1.0)], &[0.0]), 0);
    assert_eq!(inequalities(&qp, &[(0, 2, 1.0)], &[0.0]), 0);
    assert_eq!(inequalities(&qp, &[(-1, 0, 1.0)], &[0.0]), 0);
    assert_eq!(
        unsafe { PounceQpSetEqualities(qp.0, 0, null(), null(), null(), null()) },
        0
    );
    assert_eq!(
        unsafe { PounceQpSetBounds(qp.0, [f64::NAN, 0.0].as_ptr(), null()) },
        0
    );
    // Nothing was set, so the handle is still incomplete.
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_INCOMPLETE);

    // On a 1-based handle, index 0 is out of range.
    let qp = create(1, 0, 0, 1);
    assert_eq!(objective(&qp, &[(0, 0, 1.0)], &[0.0]), 0);
    assert_eq!(objective(&qp, &[(1, 1, 1.0)], &[0.0]), 1);
}

// error-path: PounceQpAddCone
#[test]
fn add_cone_refuses_bad_kinds_dims_and_exponents() {
    assert_eq!(
        unsafe { PounceQpAddCone(null_mut(), POUNCE_CONE_SOC, 3, 0.0) },
        POUNCE_QP_ERR_NULL
    );
    let qp = create(1, 0, 3, 0);
    for (kind, dim, alpha) in [
        (POUNCE_CONE_NONNEG, -1, 0.0),
        (POUNCE_CONE_SOC, 0, 0.0),
        (POUNCE_CONE_EXP, 2, 0.0),
        (POUNCE_CONE_POW, 3, 0.0),
        (POUNCE_CONE_POW, 3, 1.0),
        (POUNCE_CONE_POW, 3, f64::NAN),
        (POUNCE_CONE_POW, 4, 0.5),
        (POUNCE_CONE_PSD, 0, 0.0),
        (99, 3, 0.0),
    ] {
        assert_eq!(
            unsafe { PounceQpAddCone(qp.0, kind, dim, alpha) },
            POUNCE_QP_ERR_ARGUMENT,
            "kind {kind} dim {dim} alpha {alpha}"
        );
    }
    assert_eq!(
        unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_NONNEG, 0, 0.0) },
        0
    );
    assert_eq!(unsafe { PounceQpAddCone(qp.0, POUNCE_CONE_POW, 3, 0.5) }, 1);
}

// error-path: PounceQpSetNumOption
// error-path: PounceQpSetStrOption
#[test]
fn options_refuse_unknown_keys_and_values() {
    unsafe {
        assert_eq!(
            PounceQpSetNumOption(null_mut(), cs("tol").as_ptr(), 1e-8),
            0
        );
        assert_eq!(
            PounceQpSetStrOption(null_mut(), cs("method").as_ptr(), cs("ipm").as_ptr()),
            0
        );
    }
    let qp = create(1, 0, 0, 0);
    unsafe {
        assert_eq!(PounceQpSetNumOption(qp.0, null(), 1e-8), 0);
        assert_eq!(PounceQpSetStrOption(qp.0, cs("method").as_ptr(), null()), 0);
        assert_eq!(PounceQpSetStrOption(qp.0, null(), cs("ipm").as_ptr()), 0);
    }
    for (k, v) in [
        ("tol", 0.0),
        ("tol", 1.0),
        ("max_iter", 0.0),
        ("max_iter", 2.5),
        ("time_limit", -1.0),
        ("tau", 1.0),
        ("tau_max", 0.0),
        ("qp_reg", -1.0),
        ("bogus", 1.0),
    ] {
        assert_eq!(num_opt(&qp, k, v), 0, "{k} = {v}");
    }
    for (k, v) in [
        ("tol", 1e-9),
        ("max_iter", 50.0),
        ("time_limit", 10.0),
        ("qp_reg", 0.0),
    ] {
        assert_eq!(num_opt(&qp, k, v), 1, "{k} = {v}");
    }
    for (k, v) in [
        ("method", "simplex"),
        ("hessian_inertia", "maybe"),
        ("qp_presolve", "true"),
        ("bogus", "yes"),
    ] {
        assert_eq!(str_opt(&qp, k, v), 0, "{k} = {v}");
    }
    // max_iter = 1 stops the IPM: the option reaches the engine.
    let qp = convex_qp();
    assert_eq!(num_opt(&qp, "max_iter", 1.0), 1);
    assert_eq!(
        unsafe { PounceQpSolve(qp.0) },
        POUNCE_QP_STATUS_ITERATION_LIMIT
    );
}

// ---------------------------------------------------------------------------
// Getters: NULL handle, before any solve, after a refused solve
// ---------------------------------------------------------------------------

// error-path: PounceQpGetSolution
// error-path: PounceQpGetResiduals
// error-path: PounceQpGetCertificate
#[test]
fn getters_answer_false_without_a_result() {
    let check = |h: PounceQp| unsafe {
        let mut x = [7.0; 2];
        let mut obj = 7.0;
        let mut iters: Index = 7;
        assert_eq!(
            PounceQpGetSolution(
                h,
                x.as_mut_ptr(),
                null_mut(),
                null_mut(),
                null_mut(),
                null_mut(),
                &mut obj,
                &mut iters
            ),
            0
        );
        assert_eq!((x, obj, iters), ([7.0; 2], 7.0, 7), "nothing written");
        let mut r = PounceQpResiduals {
            kkt_error: 7.0,
            ..Default::default()
        };
        assert_eq!(PounceQpGetResiduals(h, &mut r), 0);
        assert_eq!(r.kkt_error, 7.0);
        let mut c = PounceQpCertificate {
            kind: 7,
            ..Default::default()
        };
        assert_eq!(PounceQpGetCertificate(h, &mut c), 0);
        assert_eq!(c.kind, 7);
    };
    check(null_mut());
    let qp = lp();
    check(qp.0); // before any solve
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    assert_eq!(str_opt(&qp, "qp_crossover", "yes"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_ERR_OPTIONS);
    check(qp.0); // after a refused solve

    // After a solve: NULL outputs are skipped by GetSolution, refused by the
    // struct getters.
    let qp = lp();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    unsafe {
        let mut obj = f64::NAN;
        let n = null_mut();
        assert_eq!(
            PounceQpGetSolution(qp.0, n, n, n, n, n, &mut obj, null_mut()),
            1
        );
        assert!((obj + 1.5).abs() < TOL);
        assert_eq!(PounceQpGetResiduals(qp.0, null_mut()), 0);
        assert_eq!(PounceQpGetCertificate(qp.0, null_mut()), 0);
    }
    // A builder call does not erase the result.
    assert_eq!(objective(&qp, &[], &[-1.0, -2.0]), 1);
    check_lp(&qp);
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qp_handle");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn write(qp: PounceQp, path: &std::path::Path, detail: Option<&str>) -> Bool {
    let p = cs(path.to_str().unwrap());
    let d = detail.map(cs);
    unsafe { PounceQpWriteSolveReport(qp, p.as_ptr(), d.as_ref().map_or(null(), |d| d.as_ptr())) }
}

fn read_report(path: &std::path::Path) -> pounce_solve_report::SolveReport {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

// error-path: PounceQpWriteSolveReport
#[test]
fn report_is_a_valid_solve_report_of_the_last_solve() {
    let path = tmp("lp.json");
    assert_eq!(write(null_mut(), &path, None), 0);
    let qp = lp();
    assert_eq!(write(qp.0, &path, None), 0, "no solve yet");
    assert!(!path.exists());
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    unsafe {
        assert_eq!(PounceQpWriteSolveReport(qp.0, null(), null()), 0);
    }
    assert_eq!(write(qp.0, &path, Some("verbose")), 0);
    assert_eq!(write(qp.0, &tmp("no/such/dir.json"), None), 0);

    for detail in [None, Some("summary"), Some("full")] {
        let _ = std::fs::remove_file(&path);
        assert_eq!(write(qp.0, &path, detail), 1);
        let rep = read_report(&path);
        assert_eq!(rep.schema, "pounce.solve-report/v1");
        assert_eq!(rep.problem.n_variables, 2);
        assert_eq!(rep.problem.n_constraints, 2);
        assert_eq!(rep.problem.nnz_jac_g, Some(4));
        assert_eq!(rep.problem.nnz_h_lag, Some(0));
        assert_eq!(rep.solution.engine, "cvx-qp");
        assert_eq!(rep.solution.status_upstream, "Solve_Succeeded");
        assert_eq!(rep.solution.solve_result_num, 0);
        let s = solution(&qp, 2, 1, 1);
        assert_eq!(rep.solution.x, s.x);
        assert_eq!(rep.solution.lambda, [s.y[0], s.z[0]]);
        assert_eq!(rep.solution.objective, s.obj);
        assert_eq!(rep.statistics.iteration_count, s.iters);
        let r = residuals(&qp);
        assert_eq!(rep.statistics.final_kkt_error, r.kkt_error);
        assert_eq!(rep.statistics.final_dual_inf, r.dual_infeasibility);
    }

    // The engine column follows the path: active-set, and any cone.
    let qp = lp();
    assert_eq!(str_opt(&qp, "method", "active-set"), 1);
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    assert_eq!(write(qp.0, &path, None), 1);
    assert_eq!(read_report(&path).solution.engine, "qp-active-set");
    let qp = soc();
    assert_eq!(unsafe { PounceQpSolve(qp.0) }, POUNCE_QP_STATUS_OPTIMAL);
    assert_eq!(write(qp.0, &path, None), 1);
    assert_eq!(read_report(&path).solution.engine, "cvx-qcqp");

    // A non-optimal verdict carries its status.
    let qp = create(1, 0, 0, 0);
    assert_eq!(objective(&qp, &[], &[-1.0]), 1);
    assert_eq!(
        unsafe { PounceQpSetBounds(qp.0, [0.0].as_ptr(), null()) },
        1
    );
    assert_eq!(
        unsafe { PounceQpSolve(qp.0) },
        POUNCE_QP_STATUS_DUAL_INFEASIBLE
    );
    assert_eq!(write(qp.0, &path, None), 1);
    let rep = read_report(&path);
    assert_eq!(rep.solution.status_upstream, "Diverging_Iterates");
    assert_eq!(rep.solution.solve_result_num, 300);
    let _ = std::fs::remove_file(&path);
}
