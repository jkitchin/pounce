//! Phase 2 (gh#1160): is a rank result robust?
//!
//! `numerical_rank` reports the rank of the equality Jacobian at one point,
//! under one threshold, on the unscaled matrix, from one algorithm. Each of
//! those is a choice that can manufacture or hide a rank loss, so
//! `rank_robustness` varies them one at a time:
//!
//! * **tolerance** — the rank under thresholds from 1000× below to 1000×
//!   above the default. A rank that changes within two decades of the
//!   default rests on singular values near the threshold.
//! * **scaling** — the rank after row and then column equilibration (each
//!   row, then each column, divided by its largest magnitude). A loss that
//!   appears or disappears under rescaling is about units, not structure.
//! * **algorithm** — the rank from the inertia of `[[I, Jᵀ], [J, 0]]` as
//!   pounce's sparse LDLᵀ (feral) factors it ([`feral_rank`]): an algorithm
//!   independent of the SVD that decides "zero" by a pivot test on a scaled
//!   matrix, not by a singular-value threshold. It is scale-insensitive and
//!   coarser than the SVD, so a disagreement says which way the rank is
//!   fragile. (faer's sparse QR is not rank-revealing — it orders columns for
//!   fill, not for rank — so it cannot serve here.)
//! * **points** — the rank at every point available: the main point, the
//!   model's start, the `--sol` point and each `--extra-point`. A loss present
//!   at every point is likely a property of the formulation; one present at
//!   some points is a property of those points.
//!
//! Rank loss that survives all four is reported as robust.

use super::report::{Basis, Confidence, Finding, Report, Severity, fmt_num, num};
use super::structure::{Dense, default_rel_tol, dense_jacobian_rows, equality_rows, feral_rank};
use super::{DiagnoseOptions, PointEval};
use crate::nl_reader::NlProblem;
use faer::Mat;
use pounce_common::types::Number;
use serde_json::json;

fn rank_at(s: &[Number], rel: Number) -> usize {
    let smax = s.first().copied().unwrap_or(0.0);
    s.iter().filter(|&&v| v > smax * rel).count()
}

/// Row-then-column equilibration of `d`.
fn equilibrated(d: &Dense) -> Mat<Number> {
    let (m, n) = (d.rows.len(), d.ncols());
    let mut a = Mat::from_fn(m, n, |i, j| d.a[i * n + j]);
    for i in 0..m {
        let r = (0..n).fold(0.0_f64, |x, j| x.max(a[(i, j)].abs()));
        if r > 0.0 {
            for j in 0..n {
                a[(i, j)] /= r;
            }
        }
    }
    for j in 0..n {
        let c = (0..m).fold(0.0_f64, |x, i| x.max(a[(i, j)].abs()));
        if c > 0.0 {
            for i in 0..m {
                a[(i, j)] /= c;
            }
        }
    }
    a
}

fn sorted_sv(m: &Mat<Number>) -> Option<Vec<Number>> {
    let mut s = m.singular_values().ok()?;
    s.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    Some(s)
}

/// The nonzeros of `d` as block triplets, for [`feral_rank`].
fn entries(d: &Dense) -> Vec<(usize, usize, Number)> {
    let n = d.ncols();
    let mut out = Vec::new();
    for i in 0..d.rows.len() {
        for j in 0..d.cols.len() {
            let v = d.a[i * n + j];
            if v != 0.0 {
                out.push((i, j, v));
            }
        }
    }
    out
}

/// `rank_robustness`. `others` are the further points to compare against.
pub fn check_robustness(
    prob: &NlProblem,
    main: &PointEval,
    others: &[PointEval],
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    let eq = equality_rows(prob);
    if eq.is_empty() {
        report.ran("rank_robustness");
        return;
    }
    let d = match dense_jacobian_rows(prob, main, &eq, opts) {
        Ok(d) => d,
        Err(e) => {
            report.unavailable(
                "rank_robustness",
                format!("equality Jacobian at {}: {e}", main.label),
            );
            return;
        }
    };
    let Some(s) = d.singular_values() else {
        report.unavailable("rank_robustness", "the SVD did not converge");
        return;
    };
    let m = d.rows.len();
    let rel = default_rel_tol(&d, opts);
    let rank0 = rank_at(&s, rel);
    let def0 = m - rank0;

    // Tolerance sweep.
    let factors = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3];
    let sweep: Vec<(Number, usize)> = factors
        .iter()
        .map(|f| (rel * f, m - rank_at(&s, rel * f)))
        .collect();
    let def_lo = sweep[1].1; // threshold / 100
    let def_hi = sweep[5].1; // threshold × 100
    let smax = s.first().copied().unwrap_or(0.0);
    let ratio_min = s.last().map(|v| v / smax.max(1e-300)).unwrap_or(1.0);

    // Scaling.
    let eqm = equilibrated(&d);
    let s_eq = sorted_sv(&eqm).unwrap_or_default();
    let def_scaled = m - rank_at(&s_eq, rel);

    // Algorithm.
    let feral = feral_rank(m, d.cols.len(), &entries(&d));
    let def_feral = feral.as_ref().ok().map(|r| m - r);

    // Points.
    let mut per_point: Vec<(String, Option<usize>, String)> =
        vec![(main.label.clone(), Some(def0), String::new())];
    for p in others {
        match dense_jacobian_rows(prob, p, &eq, opts).ok().and_then(|dp| {
            let sp = dp.singular_values()?;
            Some(m - rank_at(&sp, default_rel_tol(&dp, opts)))
        }) {
            Some(dp) => per_point.push((p.label.clone(), Some(dp), String::new())),
            None => per_point.push((p.label.clone(), None, "could not be evaluated".into())),
        }
    }

    report.sections.insert(
        "rank_robustness",
        json!({
            "point": main.label,
            "deficiency_at_default_threshold": def0,
            "relative_threshold": rel,
            "tolerance_sweep": sweep.iter().map(|(t, dd)| json!({"relative_threshold": t, "deficiency": dd})).collect::<Vec<_>>(),
            "sigma_min_over_sigma_max": num(ratio_min),
            "deficiency_after_equilibration": def_scaled,
            "deficiency_feral_inertia": def_feral,
            "feral_error": feral.as_ref().err(),
            "points": per_point.iter().map(|(l, dd, why)| json!({"point": l, "deficiency": dd, "note": why})).collect::<Vec<_>>(),
        }),
    );
    let mut sink = report.ran("rank_robustness");
    let mut fragile: Vec<&str> = Vec::new();
    if def_lo != def_hi {
        fragile.push("tolerance");
        sink.push(
            Finding::new("rank_robustness", "rank_tolerance_sensitive", Severity::Warning, Basis::Numerical, Confidence::High,
                format!("At {}, the equality Jacobian's rank deficiency is {def_hi} with the threshold 100× larger and {def_lo} with it 100× smaller (default: {def0}); some singular values sit within two decades of the threshold.", main.label))
            .why("Whether these rows count as dependent is a choice of tolerance, not a property of the model; nearly dependent rows still make the KKT system ill-conditioned.")
            .next("Look at the singular values in sections.numerical_rank; set --rank-tol deliberately rather than relying on the default.")
            .evidence(json!({"sweep": sweep.iter().map(|(t, dd)| json!({"relative_threshold": t, "deficiency": dd})).collect::<Vec<_>>()})),
        );
    } else if def0 == 0 && ratio_min < 1e-10 {
        sink.push(
            Finding::new("rank_robustness", "near_rank_deficient", Severity::Info, Basis::Numerical, Confidence::High,
                format!("At {}, the equality Jacobian has full rank, but its smallest singular value is {} of the largest.", main.label, fmt_num(ratio_min)))
            .why("Nearly dependent equalities make the KKT system ill-conditioned even when no tolerance calls them dependent.")
            .next("See whether equilibration (sections.rank_robustness) changes the ratio; badly scaled rows are the usual cause.")
            .evidence(json!({"sigma_min_over_sigma_max": ratio_min})),
        );
    }
    if def_scaled != def0 {
        fragile.push("scaling");
        sink.push(
            Finding::new("rank_robustness", "rank_scaling_sensitive", Severity::Warning, Basis::Numerical, Confidence::High,
                format!("At {}, the rank deficiency is {def0} unscaled and {def_scaled} after row and column equilibration.", main.label))
            .why(if def_scaled < def0 {
                "The apparent dependence comes from rows or columns of very different magnitude, not from the equations themselves."
            } else {
                "Rescaling exposes a dependence that tiny rows or columns hid below the threshold."
            })
            .next("Rescale the model's rows and variables to comparable units, then re-check."),
        );
    }
    if let Some(df) = def_feral
        && df != def0
    {
        fragile.push("algorithm");
        sink.push(
            Finding::new("rank_robustness", "rank_backend_disagreement", Severity::Info, Basis::Numerical, Confidence::Medium,
                format!("At {}, the SVD finds a rank deficiency of {def0} and the inertia of [[I, Jᵀ], [J, 0]] under pounce's sparse factorization finds {df}.", main.label))
            .why(if df > def0 {
                "The factorization treats some rows as dependent that the SVD threshold keeps: they are nearly dependent — close enough to trouble the solver's linear algebra even if not exactly so."
            } else if def_scaled < def0 {
                "The factorization works on a scaled matrix and finds rows independent that the unscaled SVD calls dependent, and the equilibrated SVD agrees with it: the SVD's loss comes from rows or columns of very different magnitude."
            } else {
                "The SVD finds the loss both unscaled and equilibrated, so the factorization's pivot test has most likely missed a dependency; trust the SVD here."
            })
            .next("Compare the tolerance sweep and the equilibrated deficiency in sections.rank_robustness.")
            .evidence(json!({"svd_deficiency": def0, "feral_inertia_deficiency": df})),
        );
    }
    let evaluated: Vec<&(String, Option<usize>, String)> =
        per_point.iter().filter(|p| p.1.is_some()).collect();
    if evaluated.len() >= 2 {
        let deficient: Vec<&str> = evaluated
            .iter()
            .filter(|p| p.1 > Some(0))
            .map(|p| p.0.as_str())
            .collect();
        if deficient.len() == evaluated.len() {
            sink.push(
                Finding::new("rank_robustness", "rank_loss_persistent", Severity::Warning, Basis::Numerical, Confidence::High,
                    format!("The equality Jacobian is rank deficient at every point checked ({}).", evaluated.iter().map(|p| format!("{}: {}", p.0, p.1.unwrap_or(0))).collect::<Vec<_>>().join("; ")))
                .why("A dependence present at unrelated points is most likely a property of the formulation, not of a point.")
                .next("Look for the redundant equation in the dependent sets of numerical_rank.")
                .evidence(json!({"points": evaluated.iter().map(|p| json!({"point": p.0, "deficiency": p.1})).collect::<Vec<_>>()})),
            );
        } else if !deficient.is_empty() {
            fragile.push("point");
            sink.push(
                Finding::new("rank_robustness", "rank_loss_point_specific", Severity::Info, Basis::Numerical, Confidence::High,
                    format!("The equality Jacobian is rank deficient at some points and not others: {}.", evaluated.iter().map(|p| format!("{}: {}", p.0, p.1.unwrap_or(0))).collect::<Vec<_>>().join("; ")))
                .why("The dependence comes from the values at particular points (often a symmetric or all-zero start), not from the formulation.")
                .next("Start from a point where the rank is full, if the model's meaning allows.")
                .evidence(json!({"points": evaluated.iter().map(|p| json!({"point": p.0, "deficiency": p.1})).collect::<Vec<_>>()})),
            );
        }
    }
    let verdict = if def0 == 0 && fragile.is_empty() {
        "full rank, robust".to_string()
    } else if fragile.is_empty() {
        format!("rank deficiency {def0}, robust to tolerance, scaling and a second algorithm")
    } else {
        format!("rank result sensitive to: {}", fragile.join(", "))
    };
    sink.push(
        Finding::new("rank_robustness", "rank_robustness_summary", Severity::Info, Basis::Numerical, Confidence::High,
            format!("At {}: {verdict}.", main.label))
        .why("Rank findings that survive changes of tolerance, scaling, algorithm and point are properties of the model; the others are properties of those choices.")
        .next("See sections.rank_robustness for the table."),
    );
}
