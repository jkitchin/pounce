//! gh#1181: the JSON report's `solution.certificate` on the convex arm.
//!
//! Each model is written out here as a linear `.nl`, and each certificate is
//! checked against *that* data — the `.nl` rows and bounds, in the support
//! function form a reader of the report would use — not against anything
//! pounce extracted from it. So the test covers the whole chain the issue
//! names: extraction, presolve, the solve, postsolve, the mapping back to
//! `.nl` constraints, and the JSON.
//!
//! One fixture per branch:
//!
//! * GALENET (netlib `infeas`), which presolve proves infeasible by itself:
//!   no ray exists to lift, so under the default `qp_certificate=yes` the
//!   certificate comes from an unreduced re-solve, and under `check` there
//!   is none; with `qp_presolve=no` it is the solve's own ray;
//! * an infeasible model with a two-sided range row, whose two halves a ray
//!   may load at once — the report must net them into one `lambda`;
//! * an unbounded **maximize** LP, where the recession direction must come
//!   back in minimize form;
//! * a feasible LP, which reports no certificate at all.

use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

const INF: f64 = f64::INFINITY;

/// A linear model in the shape the check reads: rows `l ≤ Σ a_ij x_j ≤ u`.
struct Lp {
    n: usize,
    rows: Vec<(Vec<(usize, f64)>, f64, f64)>,
    bounds: Vec<(f64, f64)>,
    obj: Vec<(usize, f64)>,
    maximize: bool,
}

/// Write `lp` as a text (`g`) `.nl`.
fn write_nl(lp: &Lp, name: &str) -> PathBuf {
    let m = lp.rows.len();
    let ranges = lp
        .rows
        .iter()
        .filter(|(_, l, u)| l.is_finite() && u.is_finite() && l != u)
        .count();
    let eqns = lp.rows.iter().filter(|(_, l, u)| l == u).count();
    let nnz: usize = lp.rows.iter().map(|(r, _, _)| r.len()).sum();
    let mut s = String::new();
    s += "g3 1 1 0\n";
    s += &format!(" {} {m} 1 {ranges} {eqns}\n", lp.n);
    s += " 0 0 0 0 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 0 0 0 0 0\n";
    s += &format!(" {nnz} {}\n", lp.obj.len());
    s += " 0 0\n 0 0 0 0 0\n";
    for i in 0..m {
        s += &format!("C{i}\nn0\n");
    }
    s += &format!("O0 {}\nn0\nx0\nr\n", u8::from(lp.maximize));
    let side = |l: f64, u: f64| -> String {
        match (l.is_finite(), u.is_finite()) {
            _ if l == u => format!("4 {l}"),
            (true, true) => format!("0 {l} {u}"),
            (false, true) => format!("1 {u}"),
            (true, false) => format!("2 {l}"),
            (false, false) => "3".to_string(),
        }
    };
    for (_, l, u) in &lp.rows {
        s += &side(*l, *u);
        s += "\n";
    }
    s += "b\n";
    for (l, u) in &lp.bounds {
        s += &side(*l, *u);
        s += "\n";
    }
    // Cumulative column counts of the Jacobian, first n − 1 columns.
    let mut per_col = vec![0usize; lp.n];
    for (r, _, _) in &lp.rows {
        for (j, _) in r {
            per_col[*j] += 1;
        }
    }
    s += &format!("k{}\n", lp.n - 1);
    let mut acc = 0;
    for c in per_col.iter().take(lp.n - 1) {
        acc += c;
        s += &format!("{acc}\n");
    }
    for (i, (r, _, _)) in lp.rows.iter().enumerate() {
        s += &format!("J{i} {}\n", r.len());
        for (j, a) in r {
            s += &format!("{j} {a}\n");
        }
    }
    if !lp.obj.is_empty() {
        s += &format!("G0 {}\n", lp.obj.len());
        for (j, c) in &lp.obj {
            s += &format!("{j} {c}\n");
        }
    }
    let dir = std::env::temp_dir().join(format!("pounce-gh1181-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.nl"));
    std::fs::write(&path, s).unwrap();
    path
}

/// Solve and return `(stdout, solution)` from the JSON report.
fn solve(nl: &PathBuf, extra: &[&str]) -> (String, Value) {
    let json = nl.with_extension(format!("{}.json", extra.join("_").replace('=', "-")));
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(nl)
        .args(["--no-sol", "--no-options-file", "--json-output"])
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let report: Value = serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        report["solution"].clone(),
    )
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

/// `lambda` (positive on a row's upper side, negative on its lower), `z_l`,
/// `z_u` prove the rows and bounds empty: `Σ λᵢ aᵢ − z_l + z_u = 0` and the
/// support value `Σ λᵢ·(uᵢ or lᵢ) − lᵀz_l + uᵀz_u < 0`.
fn check_farkas(lp: &Lp, cert: &Value) {
    assert_eq!(cert["kind"], "primal_infeasible", "{cert}");
    assert_eq!(cert["verified"], true, "{cert}");
    let (lam, zl, zu) = (
        floats(&cert["lambda"]),
        floats(&cert["z_l"]),
        floats(&cert["z_u"]),
    );
    assert_eq!(lam.len(), lp.rows.len());
    let mut r = vec![0.0; lp.n];
    let mut value = 0.0;
    for ((row, l, u), &li) in lp.rows.iter().zip(&lam) {
        if li > 0.0 {
            assert!(u.is_finite(), "lambda {li} on an infinite upper side");
            value += li * u;
        } else if li < 0.0 {
            assert!(l.is_finite(), "lambda {li} on an infinite lower side");
            value += li * l;
        }
        for (j, a) in row {
            r[*j] += a * li;
        }
    }
    for (j, (l, u)) in lp.bounds.iter().enumerate() {
        assert!(zl[j] >= 0.0 && zu[j] >= 0.0);
        if zl[j] > 0.0 {
            assert!(l.is_finite());
            value -= l * zl[j];
        }
        if zu[j] > 0.0 {
            assert!(u.is_finite());
            value += u * zu[j];
        }
        r[j] += zu[j] - zl[j];
    }
    assert!(r.iter().all(|v| v.abs() <= 1e-9), "residual {r:?}");
    assert!(value < -1e-6, "value {value}");
    let reported = cert["value"].as_f64().unwrap();
    assert!(
        (reported - value).abs() <= 1e-9 * value.abs().max(1.0),
        "{reported} vs {value}"
    );
}

/// GALENET (netlib `infeas`), rows in the netlib order: supplies `S1..S3 ≤
/// 20`, transshipment `NODE4`, `NODE5` `= 0`, demands `D6 ≥ 10`, `D7 ≥ 20`,
/// `D8 ≥ 30`; columns `T14 T24 T25 T35 T46 T47 T57 T58`. Node 5 can take in
/// at most 20 and must send out 30.
fn galenet() -> Lp {
    Lp {
        n: 8,
        rows: vec![
            (vec![(0, 1.0)], -INF, 20.0),
            (vec![(1, 1.0), (2, 1.0)], -INF, 20.0),
            (vec![(3, 1.0)], -INF, 20.0),
            (vec![(0, 1.0), (1, 1.0), (4, -1.0), (5, -1.0)], 0.0, 0.0),
            (vec![(2, 1.0), (3, 1.0), (6, -1.0), (7, -1.0)], 0.0, 0.0),
            (vec![(4, 1.0)], 10.0, INF),
            (vec![(5, 1.0), (6, 1.0)], 20.0, INF),
            (vec![(7, 1.0)], 30.0, INF),
        ],
        bounds: [30.0, 20.0, 10.0, 10.0, 10.0, 2.0, 20.0, 30.0]
            .iter()
            .map(|&u| (0.0, u))
            .collect(),
        obj: vec![],
        maximize: false,
    }
}

#[test]
fn galenet_reports_a_checked_farkas_certificate() {
    let lp = galenet();
    let nl = write_nl(&lp, "galenet");

    // Presolve proves it on its own; the ray comes from the re-solve.
    let (out, sol) = solve(&nl, &[]);
    assert_eq!(sol["status"], "InfeasibleProblemDetected", "{sol}");
    assert_eq!(sol["certificate"]["source"], "unreduced_resolve");
    check_farkas(&lp, &sol["certificate"]);
    assert!(
        out.lines()
            .any(|l| l.starts_with("Certificate: Farkas") && l.contains("verified")),
        "{out}"
    );

    // Without presolve the verdict's own ray is the certificate.
    let (_, sol) = solve(&nl, &["qp_presolve=no"]);
    assert_eq!(sol["certificate"]["source"], "solve");
    check_farkas(&lp, &sol["certificate"]);

    // `check` never re-solves: a presolve verdict then has no certificate.
    let (out, sol) = solve(&nl, &["qp_certificate=check"]);
    assert_eq!(sol["status"], "InfeasibleProblemDetected");
    assert!(sol.get("certificate").is_none(), "{sol}");
    assert!(!out.contains("Certificate:"), "{out}");

    let (_, sol) = solve(&nl, &["qp_presolve=no", "qp_certificate=no"]);
    assert!(sol.get("certificate").is_none(), "{sol}");
}

/// `1 ≤ x₀ + x₁ ≤ 2` (one range row, two rows once extracted) beside
/// `x₀ + 2x₁ ≥ 5` over `[0, 1]²`, where the second row's largest activity is
/// 3. `lambda` must be one number per `.nl` row.
#[test]
fn a_range_row_reports_one_netted_multiplier() {
    let lp = Lp {
        n: 2,
        rows: vec![
            (vec![(0, 1.0), (1, 1.0)], 1.0, 2.0),
            (vec![(0, 1.0), (1, 2.0)], 5.0, INF),
        ],
        bounds: vec![(0.0, 1.0), (0.0, 1.0)],
        obj: vec![(0, 1.0)],
        maximize: false,
    };
    let nl = write_nl(&lp, "range_row");
    for extra in [&[][..], &["qp_presolve=no"][..]] {
        let (_, sol) = solve(&nl, extra);
        assert_eq!(
            sol["status"], "InfeasibleProblemDetected",
            "{extra:?}: {sol}"
        );
        check_farkas(&lp, &sol["certificate"]);
    }
}

/// `max x₀ + x₁` over `x₀ − x₁ = 0`, `x₀ + x₁ ≥ 1`, `x ≥ 0`: unbounded
/// along `d = (1, 1)`. The report is in minimize form, so `value = −2/‖d‖`.
#[test]
fn an_unbounded_maximize_lp_reports_its_recession_direction() {
    let lp = Lp {
        n: 2,
        rows: vec![
            (vec![(0, 1.0), (1, -1.0)], 0.0, 0.0),
            (vec![(0, 1.0), (1, 1.0)], 1.0, INF),
        ],
        bounds: vec![(0.0, INF), (0.0, INF)],
        obj: vec![(0, 1.0), (1, 1.0)],
        maximize: true,
    };
    let nl = write_nl(&lp, "unbounded_max");
    for extra in [&[][..], &["qp_presolve=no"][..]] {
        let (out, sol) = solve(&nl, extra);
        assert_eq!(sol["status"], "DivergingIterates", "{extra:?}: {sol}");
        let cert = &sol["certificate"];
        assert_eq!(cert["kind"], "dual_infeasible", "{cert}");
        assert_eq!(cert["verified"], true, "{cert}");
        let d = floats(&cert["d"]);
        // Feasible direction: the equality holds, the `≥` row does not fall,
        // the bounds are not crossed, and the user's objective rises.
        assert!((d[0] - d[1]).abs() <= 1e-9, "{d:?}");
        assert!(
            d[0] + d[1] >= -1e-9 && d.iter().all(|&v| v >= -1e-9),
            "{d:?}"
        );
        assert!(d[0] + d[1] > 1e-6, "{d:?}");
        assert!(
            cert["value"].as_f64().unwrap() < 0.0,
            "minimize form: {cert}"
        );
        assert!(out.contains("Certificate: recession"), "{out}");
    }
}

#[test]
fn a_feasible_lp_reports_no_certificate() {
    let lp = Lp {
        n: 2,
        rows: vec![(vec![(0, 1.0), (1, 1.0)], 1.0, INF)],
        bounds: vec![(0.0, INF), (0.0, INF)],
        obj: vec![(0, 1.0), (1, 2.0)],
        maximize: false,
    };
    let nl = write_nl(&lp, "feasible");
    let (out, sol) = solve(&nl, &[]);
    assert_eq!(sol["status"], "SolveSucceeded", "{sol}");
    assert!(sol.get("certificate").is_none(), "{sol}");
    assert!(!out.contains("Certificate:"), "{out}");
}
