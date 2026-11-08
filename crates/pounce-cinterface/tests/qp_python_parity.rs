//! Parity of the `PounceQp*` entry with Python's `solve_qp` / `solve_socp`
//! (gh#1147 S3-I3).
//!
//! `common/qp_python.json` (written by `common/gen_qp_python.py`) holds, per
//! row, the exact `QpProblem` Python's frontend hands its binding — the
//! triplets in `_lower_triangle_coo` / `_coo` order, `b`, `h`, `lb`, `ub`, the
//! cone list — the keyword options, and Python's result. Each row is fed to
//! the C handle (same triplets, same order, the options spelled the C way)
//! and compared three ways:
//!
//! 1. **Python, bit-exact** — status, `obj`, `iters`, `x`, `y`, `z`, `z_lb`,
//!    `z_ub`, all eight residuals, `tau`, `kappa` and `certificate_scale`.
//!    Both sides run the same `pounce_convex::pipeline` code on the same
//!    `QpProblem`, so nothing may differ. The one difference in the inputs
//!    is the box: Python passes an omitted `lb`/`ub` as an empty vector, the
//!    handle as `n` infinities; `pounce_convex` reads both as "no bound"
//!    (`QpProblem::lb_of`), and the bitwise match on every row with an
//!    omitted side (all the conic rows, `qp_primal_infeasible`,
//!    `qp_dual_infeasible`, `qp_coupled`, the LP rows' `ub`) is the evidence.
//!    A checked-in floating-point reference is exact only on the platform it
//!    was generated on (Linux x86_64: libm and the FMA contraction differ
//!    elsewhere), so the bitwise comparison is gated on that target. Elsewhere
//!    the row still has to reach Python's status and, where the status is a
//!    solution, Python's `x` and `obj` to `1e-6`.
//! 2. **`pounce_convex` directly, bit-exact on every platform** — the same
//!    `QpProblem` Python builds (empty box sides included), through the
//!    `pounce_convex` functions the row's path selects. Same binary, so no
//!    platform gate: this is §7's "same iteration counts, since it is the
//!    same pipeline".
//! 3. **The hand-known answer** of each row (the table in `known`), to a
//!    tolerance, so a reference regenerated from a broken build cannot pass
//!    by agreeing with itself. The conic rows' models are Python's own test
//!    anchors, cited by file and line there.
//!
//! **Branches reached** (CLAUDE.md, "A test is only evidence about the branch
//! it reaches"): the plain IPM on the HSDE driver and the direct driver; the
//! presolved IPM with crossover (presolve settles the LP: 0 iterations); the
//! active-set engine plain, presolved, and on an indefinite `P`; `tau`/
//! `tau_max`, `qp_reg`, `qp_equilibrate=no`, `tol`, `max_iter`; every cone
//! kind (`nonneg`, `soc`, `exp`, `pow`, `psd`) and mixes on both conic drivers
//! (symmetric, and the non-symmetric one for exp/pow, which has no `tau`); and
//! on both the QP and the conic path the statuses `optimal`,
//! `iteration_limit`, `primal_infeasible` and `dual_infeasible`, so both
//! certificate kinds and `certificate_scale` are compared. Every row asserts
//! its status, so a model that drifts off its branch fails here.
//!
//! The PSD screen differs between the frontends (Python's numpy eigenvalue
//! verdict, the handle's `certify_psd_lower_triangle`); every `P` here is far
//! from the PSD border (eigenvalues 1, 2, 3 or −1), where both agree.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/qp_c.rs"]
mod qp_c;

use pounce_cinterface::*;
use pounce_convex::pipeline::{EngineSwitches, demote_unverified_optimal, solve_presolved};
use pounce_convex::{
    ActiveSetOverrides, ConeSpec, HessianInertia, QpOptions, QpProblem, QpSolution, Triplet,
    solve_qp_active_set_inertia, solve_qp_ipm, solve_socp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use qp_c::{Qp, TRUE};
use serde_json::Value;

const PYTHON: &str = include_str!("common/qp_python.json");

const EXACT_PLATFORM: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn rows() -> serde_json::Map<String, Value> {
    let doc: Value = serde_json::from_str(PYTHON).expect("parse the Python reference");
    doc["rows"].as_object().unwrap().clone()
}

fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// A bound vector: null is the infinity of `inf`'s sign.
fn box_side(v: &Value, inf: f64) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(inf))
        .collect()
}

fn index(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn triplets(v: &Value) -> Vec<(usize, usize, f64)> {
    let (r, c, x) = (index(&v[0]), index(&v[1]), reals(&v[2]));
    assert!(r.len() == c.len() && c.len() == x.len());
    (0..r.len()).map(|k| (r[k], c[k], x[k])).collect()
}

fn cones(model: &Value) -> Vec<ConeSpec> {
    model["cones"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let v = c[1].as_f64().unwrap();
            match c[0].as_str().unwrap() {
                "nonneg" => ConeSpec::Nonneg(v as usize),
                "soc" => ConeSpec::SecondOrder(v as usize),
                "exp" => ConeSpec::Exponential,
                "pow" => ConeSpec::Power(v),
                "psd" => ConeSpec::Psd(v as usize),
                k => panic!("cone kind {k}"),
            }
        })
        .collect()
}

/// The `QpProblem` Python's `_build` makes: omitted box sides stay empty.
fn python_problem(m: &Value) -> QpProblem {
    let t = |v: &Value| {
        triplets(v)
            .into_iter()
            .map(|(r, c, x)| Triplet::new(r, c, x))
            .collect::<Vec<_>>()
    };
    QpProblem {
        n: m["n"].as_u64().unwrap() as usize,
        p_lower: t(&m["p"]),
        c: reals(&m["c"]),
        a: t(&m["a"]),
        b: reals(&m["b"]),
        g: t(&m["g"]),
        h: reals(&m["h"]),
        lb: box_side(&m["lb"], f64::NEG_INFINITY),
        ub: box_side(&m["ub"], f64::INFINITY),
    }
}

fn yes_no(v: &Value) -> &'static str {
    if v.as_bool().unwrap() { "yes" } else { "no" }
}

/// Build the row's model on a C handle (0-based) and set its options.
fn c_handle(m: &Value, opts: &serde_json::Map<String, Value>) -> Qp {
    let dim = |k: &str| m[k].as_u64().unwrap() as usize;
    let qp = Qp::new(dim("n"), dim("m_eq"), dim("m_ineq"), 0);
    let t = |v: &Value| {
        triplets(v)
            .into_iter()
            .map(|(r, c, x)| (r as Index, c as Index, x))
            .collect::<Vec<_>>()
    };
    assert_eq!(qp.objective(&t(&m["p"]), &reals(&m["c"])), TRUE);
    if qp.m_eq > 0 {
        assert_eq!(qp.equalities(&t(&m["a"]), &reals(&m["b"])), TRUE);
    }
    if qp.m_ineq > 0 {
        assert_eq!(qp.inequalities(&t(&m["g"]), &reals(&m["h"])), TRUE);
    }
    let lb = box_side(&m["lb"], f64::NEG_INFINITY);
    let ub = box_side(&m["ub"], f64::INFINITY);
    let side = |v: &[f64]| (!v.is_empty()).then_some(v.to_vec());
    let (lb, ub) = (side(&lb), side(&ub));
    if lb.is_some() || ub.is_some() {
        assert_eq!(qp.bounds(lb.as_deref(), ub.as_deref()), TRUE);
    }
    for (k, c) in cones(m).iter().enumerate() {
        let (kind, d, alpha) = match *c {
            ConeSpec::Nonneg(d) => (POUNCE_CONE_NONNEG, d, 0.0),
            ConeSpec::SecondOrder(d) => (POUNCE_CONE_SOC, d, 0.0),
            ConeSpec::Exponential => (POUNCE_CONE_EXP, 3, 0.0),
            ConeSpec::Power(a) => (POUNCE_CONE_POW, 3, a),
            ConeSpec::Psd(d) => (POUNCE_CONE_PSD, d, 0.0),
        };
        assert_eq!(qp.cone(kind, d as Index, alpha), k as i32);
    }
    for (k, v) in opts {
        let ok = match k.as_str() {
            "method" => qp.str_opt("method", v.as_str().unwrap()),
            "qp_presolve" | "qp_hsde" | "qp_equilibrate" | "qp_crossover" => {
                qp.str_opt(k, yes_no(v))
            }
            "tol" | "max_iter" | "tau" | "tau_max" | "qp_reg" => qp.num_opt(k, v.as_f64().unwrap()),
            other => panic!("no C spelling for option {other}"),
        };
        assert_eq!(ok, TRUE, "option {k} = {v}");
    }
    qp
}

/// The `pounce_convex` calls the row's path makes, on Python's `QpProblem`.
fn direct(entry: &str, m: &Value, opts: &serde_json::Map<String, Value>) -> QpSolution {
    let prob = python_problem(m);
    let num = |k: &str| opts.get(k).and_then(Value::as_f64);
    let flag = |k: &str| opts.get(k).map(|v| v.as_bool().unwrap());
    let mut o = QpOptions::default();
    if let Some(t) = num("tol") {
        o.tol = t;
    }
    if let Some(n) = num("max_iter") {
        o.max_iter = n as usize;
    }
    if let Some(t) = num("tau") {
        o.tau = t;
        o.tau_max = o.tau_max.max(t);
    }
    if let Some(t) = num("tau_max") {
        o.tau_max = t;
    }
    let sw = EngineSwitches {
        presolve: flag("qp_presolve"),
        reg: num("qp_reg"),
        hsde: flag("qp_hsde"),
        equilibrate: flag("qp_equilibrate"),
        crossover: flag("qp_crossover"),
    };
    sw.apply(&mut o);
    let presolve = sw.presolve.unwrap_or(false);
    if entry == "socp" {
        return solve_socp_ipm(&prob, &cones(m), &o, backend);
    }
    match opts.get("method").and_then(Value::as_str).unwrap_or("ipm") {
        "ipm" if presolve => {
            let (sol, _) = solve_presolved(&prob, &o, |p, po| solve_qp_ipm(p, po, backend));
            demote_unverified_optimal(&prob, &o, sol)
        }
        "ipm" => solve_qp_ipm(&prob, &o, backend),
        "active-set" => {
            // Python's verdict on these models: `indefinite_active_set` is the
            // one indefinite `P`.
            let inertia = if prob.p_lower.iter().any(|t| t.val < 0.0 && t.row == t.col) {
                HessianInertia::Indefinite
            } else {
                HessianInertia::Psd
            };
            let ov = ActiveSetOverrides::default();
            let mut mk = backend;
            if presolve {
                solve_presolved(&prob, &o, |p, po| {
                    solve_qp_active_set_inertia(p, po, &ov, inertia, &mut mk)
                })
                .0
            } else {
                solve_qp_active_set_inertia(&prob, &o, &ov, inertia, &mut mk)
            }
        }
        other => panic!("method {other}"),
    }
}

fn status_code(s: &str) -> i32 {
    match s {
        "optimal" => POUNCE_QP_STATUS_OPTIMAL,
        "optimal_inaccurate" => POUNCE_QP_STATUS_OPTIMAL_INACCURATE,
        "primal_infeasible" => POUNCE_QP_STATUS_PRIMAL_INFEASIBLE,
        "dual_infeasible" => POUNCE_QP_STATUS_DUAL_INFEASIBLE,
        "iteration_limit" => POUNCE_QP_STATUS_ITERATION_LIMIT,
        "time_limit" => POUNCE_QP_STATUS_TIME_LIMIT,
        "numerical_failure" => POUNCE_QP_STATUS_NUMERICAL_FAILURE,
        s => panic!("status {s}"),
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Python's optional real: null is the C side's NaN.
fn opt_bits(v: &Value) -> u64 {
    v.as_f64().unwrap_or(f64::NAN).to_bits()
}

/// Each row's hand-known answer: `(status, Some((obj, x)))` for a solution,
/// `None` for a status that carries none.
fn known(name: &str) -> (i32, Option<(f64, Vec<f64>)>) {
    let opt = POUNCE_QP_STATUS_OPTIMAL;
    let r2 = std::f64::consts::FRAC_1_SQRT_2;
    let e = std::f64::consts::E;
    let pow = 2f64.powf(0.3) * 0.5f64.powf(0.7);
    match name {
        "lp_ipm" | "lp_presolve_crossover" | "lp_direct" | "lp_active_set" => {
            (opt, Some((-1.5, vec![0.5, 0.5])))
        }
        "qp_ipm" | "qp_ipm_tau" | "qp_ipm_reg_no_equilibrate" | "qp_active_set_presolve" => {
            (opt, Some((-0.75, vec![0.5, 0.5])))
        }
        "qp_coupled" => (opt, Some((-1.0, vec![1.0, 1.0]))),
        // Either local minimum (0, ±1); the sign is checked as |x1| below.
        "indefinite_active_set" => (opt, Some((-0.5, vec![0.0, -1.0]))),
        "qp_iteration_limit" | "soc_iteration_limit" => (POUNCE_QP_STATUS_ITERATION_LIMIT, None),
        "qp_primal_infeasible" | "soc_primal_infeasible" => {
            (POUNCE_QP_STATUS_PRIMAL_INFEASIBLE, None)
        }
        "qp_dual_infeasible" | "soc_dual_infeasible" => (POUNCE_QP_STATUS_DUAL_INFEASIBLE, None),
        "nonneg" => (opt, Some((-3.0, vec![1.0, 2.0]))),
        "soc_distance" | "soc_distance_tol" => (opt, Some((r2, vec![r2, 1.5, -1.5]))),
        "soc_projection" => (opt, Some((-2.25, vec![1.5, 1.5, 0.0]))),
        "nonneg_and_soc" => (opt, Some((-2.0, vec![1.0, 1.0]))),
        "exp_geometric_program" => (opt, Some((2.0, vec![0.0, 1.0, 1.0]))),
        "soc_and_exp" => (opt, Some((5.0 + e, vec![5.0, e]))),
        "power" => (opt, Some((-pow, vec![pow, 2.0, 0.5]))),
        "psd" => (opt, Some((-1.0, vec![1.0]))),
        other => panic!("row {other} has no hand-known answer; add one"),
    }
}

/// Every row the generator writes, so a row dropped from the reference (or
/// added without an answer here) fails.
const ROWS: [&str; 25] = [
    "lp_ipm",
    "lp_presolve_crossover",
    "lp_direct",
    "lp_active_set",
    "qp_ipm",
    "qp_ipm_tau",
    "qp_ipm_reg_no_equilibrate",
    "qp_active_set_presolve",
    "qp_coupled",
    "indefinite_active_set",
    "qp_iteration_limit",
    "qp_primal_infeasible",
    "qp_dual_infeasible",
    "nonneg",
    "soc_distance",
    "soc_distance_tol",
    "soc_projection",
    "nonneg_and_soc",
    "exp_geometric_program",
    "soc_and_exp",
    "power",
    "psd",
    "soc_primal_infeasible",
    "soc_dual_infeasible",
    "soc_iteration_limit",
];

#[test]
fn reference_has_exactly_the_listed_rows() {
    let mut have: Vec<String> = rows().keys().cloned().collect();
    have.sort();
    let mut want: Vec<String> = ROWS.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(have, want);
}

fn run_row(name: &str) {
    let all = rows();
    let row = all
        .get(name)
        .unwrap_or_else(|| panic!("no Python row {name}"));
    let (entry, m, py) = (
        row["entry"].as_str().unwrap(),
        &row["model"],
        &row["result"],
    );
    let opts = row["options"].as_object().unwrap();

    let qp = c_handle(m, opts);
    let code = qp.solve();
    let s = qp.solution();
    let res = qp.residuals();
    let cert = qp.certificate();

    // 3. The hand-known answer.
    let (want_status, want) = known(name);
    assert_eq!(code, want_status, "{name}: status against the known answer");
    if let Some((obj, x)) = &want {
        assert!(
            (s.obj - *obj).abs() < 1e-5,
            "{name}: obj {} vs {obj}",
            s.obj
        );
        let xs: Vec<f64> = if name == "indefinite_active_set" {
            vec![s.x[0], -s.x[1].abs()]
        } else {
            s.x.clone()
        };
        qp_c::close(&format!("{name}: x"), &xs, x, 1e-4);
    }

    // 2. pounce_convex directly, on Python's QpProblem: bitwise everywhere.
    let d = direct(entry, m, opts);
    assert_eq!(code, status_code_of(d.status), "{name}: direct status");
    assert_eq!(s.iters as usize, d.iters, "{name}: direct iterations");
    assert_eq!(s.obj.to_bits(), d.obj.to_bits(), "{name}: direct obj");
    for (what, c, r) in [
        ("x", &s.x, &d.x),
        ("y", &s.y, &d.y),
        ("z", &s.z, &d.z),
        ("z_lb", &s.z_lb, &d.z_lb),
        ("z_ub", &s.z_ub, &d.z_ub),
    ] {
        assert_eq!(bits(c), bits(r), "{name}: direct {what}");
    }

    // 1. Python.
    assert_eq!(
        code,
        status_code(py["status"].as_str().unwrap()),
        "{name}: status against Python"
    );
    if !EXACT_PLATFORM {
        eprintln!(
            "{name}: Python bitwise parity not compared: the reference was generated on \
             Linux x86_64 and is exact only there (status, x and obj compared to 1e-6; \
             the direct pounce_convex comparison above is still bitwise)"
        );
        if want.is_some() {
            assert!(
                (s.obj - py["obj"].as_f64().unwrap()).abs() < 1e-6,
                "{name}: obj"
            );
            qp_c::close(&format!("{name}: x"), &s.x, &reals(&py["x"]), 1e-6);
        }
        return;
    }
    assert_eq!(
        s.iters as u64,
        py["iters"].as_u64().unwrap(),
        "{name}: iters"
    );
    assert_eq!(
        s.obj.to_bits(),
        py["obj"].as_f64().unwrap().to_bits(),
        "{name}: obj {} vs {}",
        s.obj,
        py["obj"]
    );
    for (what, c) in [
        ("x", &s.x),
        ("y", &s.y),
        ("z", &s.z),
        ("z_lb", &s.z_lb),
        ("z_ub", &s.z_ub),
    ] {
        assert_eq!(
            bits(c),
            bits(&reals(&py[what])),
            "{name}: {what} {c:?} vs {}",
            py[what]
        );
    }
    let pr = &py["residuals"];
    for (what, c) in [
        ("primal_infeasibility", res.primal_infeasibility),
        ("dual_infeasibility", res.dual_infeasibility),
        ("complementarity", res.complementarity),
        ("kkt_error", res.kkt_error),
        ("primal_infeasibility_raw", res.primal_infeasibility_raw),
        ("dual_infeasibility_raw", res.dual_infeasibility_raw),
        ("complementarity_raw", res.complementarity_raw),
        ("kkt_error_raw", res.kkt_error_raw),
    ] {
        assert_eq!(
            c.to_bits(),
            pr[what].as_f64().unwrap().to_bits(),
            "{name}: residuals.{what}"
        );
    }
    assert_eq!(cert.tau.to_bits(), opt_bits(&py["tau"]), "{name}: tau");
    assert_eq!(
        cert.kappa.to_bits(),
        opt_bits(&py["kappa"]),
        "{name}: kappa"
    );
    assert_eq!(
        cert.scale.to_bits(),
        opt_bits(&py["certificate_scale"]),
        "{name}: certificate_scale"
    );
    let kind = match code {
        POUNCE_QP_STATUS_PRIMAL_INFEASIBLE => POUNCE_QP_CERT_PRIMAL_INFEASIBLE,
        POUNCE_QP_STATUS_DUAL_INFEASIBLE => POUNCE_QP_CERT_DUAL_INFEASIBLE,
        _ => POUNCE_QP_CERT_NONE,
    };
    assert_eq!(cert.kind, kind, "{name}: certificate kind");
}

fn status_code_of(s: pounce_convex::QpStatus) -> i32 {
    use pounce_convex::QpStatus::*;
    match s {
        Optimal => POUNCE_QP_STATUS_OPTIMAL,
        OptimalInaccurate => POUNCE_QP_STATUS_OPTIMAL_INACCURATE,
        PrimalInfeasible => POUNCE_QP_STATUS_PRIMAL_INFEASIBLE,
        DualInfeasible => POUNCE_QP_STATUS_DUAL_INFEASIBLE,
        IterationLimit => POUNCE_QP_STATUS_ITERATION_LIMIT,
        TimeLimit => POUNCE_QP_STATUS_TIME_LIMIT,
        NumericalFailure => POUNCE_QP_STATUS_NUMERICAL_FAILURE,
    }
}

macro_rules! rows {
    ($($t:ident),* $(,)?) => { $( #[test] fn $t() { run_row(stringify!($t)); } )* };
}

rows!(
    lp_ipm,
    lp_presolve_crossover,
    lp_direct,
    lp_active_set,
    qp_ipm,
    qp_ipm_tau,
    qp_ipm_reg_no_equilibrate,
    qp_active_set_presolve,
    qp_coupled,
    indefinite_active_set,
    qp_iteration_limit,
    qp_primal_infeasible,
    qp_dual_infeasible,
    nonneg,
    soc_distance,
    soc_distance_tol,
    soc_projection,
    nonneg_and_soc,
    exp_geometric_program,
    soc_and_exp,
    power,
    psd,
    soc_primal_infeasible,
    soc_dual_infeasible,
    soc_iteration_limit,
);
