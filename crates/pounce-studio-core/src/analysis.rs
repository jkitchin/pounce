//! Derived series and diagnostics over a [`SolveReport`].
//!
//! Mirrors the Python analysis helpers in `studio/mcp/pounce_studio_mcp/
//! reports.py` so the desktop / VS Code shells and the MCP server can
//! agree on the same notion of "stall window", "restoration window",
//! and "common failure modes". Heuristics are tunable via the
//! parameters on each function; the defaults are the ones the Python
//! `diagnose` tool ships with.

use serde::{Deserialize, Serialize};

use crate::report::{Error, IterRecord, SolveReport};

/// Compact view-model derived from a [`SolveReport`]. Suitable as the
/// "headline summary" that an LLM or dashboard reads first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub schema: String,
    pub result_id: String,
    pub solver: String,
    pub solver_version: String,
    pub elapsed_seconds: f64,
    pub n_variables: i32,
    pub n_constraints: i32,
    pub status: String,
    pub final_objective: f64,
    pub iteration_count: i32,
    pub final_kkt_error: f64,
    pub final_dual_inf: f64,
    pub final_constr_viol: f64,
    pub final_compl: f64,
    pub restoration_calls: i32,
    pub restoration_outer_iters: i32,
    pub restoration_wall_secs: f64,
    pub iterations_captured: usize,
}

pub fn summarize(report: &SolveReport) -> Summary {
    Summary {
        schema: report.schema.clone(),
        result_id: report.fair_metadata.result_id.clone(),
        solver: report.fair_metadata.solver.name.clone(),
        solver_version: report.fair_metadata.solver.version.clone(),
        elapsed_seconds: report.fair_metadata.elapsed_seconds,
        n_variables: report.problem.n_variables,
        n_constraints: report.problem.n_constraints,
        status: report.solution.status.clone(),
        final_objective: report.statistics.final_objective,
        iteration_count: report.statistics.iteration_count,
        final_kkt_error: report.statistics.final_kkt_error,
        final_dual_inf: report.statistics.final_dual_inf,
        final_constr_viol: report.statistics.final_constr_viol,
        final_compl: report.statistics.final_compl,
        restoration_calls: report.statistics.restoration_calls,
        restoration_outer_iters: report.statistics.restoration_outer_iters,
        restoration_wall_secs: report.statistics.restoration_wall_secs,
        iterations_captured: report.iterations.len(),
    }
}

/// Per-iteration trajectory in column-oriented form. More compact than
/// a `Vec<IterRecord>` when serialised, since the column names are
/// emitted once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceTrace {
    pub iter: Vec<i32>,
    pub objective: Vec<f64>,
    pub inf_pr: Vec<f64>,
    pub inf_du: Vec<f64>,
    pub mu: Vec<f64>,
    pub d_norm: Vec<f64>,
    pub regularization: Vec<f64>,
    pub alpha_dual: Vec<f64>,
    pub alpha_primal: Vec<f64>,
    pub alpha_primal_char: Vec<char>,
    pub ls_trials: Vec<i32>,
}

pub fn convergence_trace(report: &SolveReport) -> ConvergenceTrace {
    let n = report.iterations.len();
    let mut t = ConvergenceTrace {
        iter: Vec::with_capacity(n),
        objective: Vec::with_capacity(n),
        inf_pr: Vec::with_capacity(n),
        inf_du: Vec::with_capacity(n),
        mu: Vec::with_capacity(n),
        d_norm: Vec::with_capacity(n),
        regularization: Vec::with_capacity(n),
        alpha_dual: Vec::with_capacity(n),
        alpha_primal: Vec::with_capacity(n),
        alpha_primal_char: Vec::with_capacity(n),
        ls_trials: Vec::with_capacity(n),
    };
    for r in &report.iterations {
        t.iter.push(r.iter);
        t.objective.push(r.objective);
        t.inf_pr.push(r.inf_pr);
        t.inf_du.push(r.inf_du);
        t.mu.push(r.mu);
        t.d_norm.push(r.d_norm);
        t.regularization.push(r.regularization);
        t.alpha_dual.push(r.alpha_dual);
        t.alpha_primal.push(r.alpha_primal);
        t.alpha_primal_char.push(r.alpha_primal_char);
        t.ls_trials.push(r.ls_trials);
    }
    t
}

/// One stalled-progress window: consecutive iterations whose
/// log10-residual moved by less than the configured threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stall {
    pub start_iter: i32,
    pub end_iter: i32,
    pub metric: &'static str,
    pub delta_log10: f64,
}

/// Default stall detection: 5+ consecutive iters with <0.3 orders of
/// magnitude movement in either `inf_pr` or `inf_du`.
pub fn find_stalls(report: &SolveReport) -> Vec<Stall> {
    find_stalls_with(report, 5, 0.3)
}

pub fn find_stalls_with(
    report: &SolveReport,
    min_window: usize,
    max_log10_progress: f64,
) -> Vec<Stall> {
    let mut out = Vec::new();
    for (metric, series) in [
        ("inf_pr", series_log10(&report.iterations, |r| r.inf_pr)),
        ("inf_du", series_log10(&report.iterations, |r| r.inf_du)),
    ] {
        scan_stalls(
            &series,
            &report.iterations,
            metric,
            min_window,
            max_log10_progress,
            &mut out,
        );
    }
    out
}

fn series_log10<F: Fn(&IterRecord) -> f64>(iters: &[IterRecord], f: F) -> Vec<Option<f64>> {
    iters
        .iter()
        .map(|r| {
            let v = f(r);
            if v > 0.0 && v.is_finite() {
                Some(v.log10())
            } else {
                None
            }
        })
        .collect()
}

fn scan_stalls(
    series: &[Option<f64>],
    iters: &[IterRecord],
    metric: &'static str,
    min_window: usize,
    max_log10_progress: f64,
    out: &mut Vec<Stall>,
) {
    let mut i = 0;
    let n = series.len();
    while i < n {
        if series[i].is_none() {
            i += 1;
            continue;
        }
        // Greedy: extend j while [i..=j] remains a stall.
        let mut j = i;
        let mut win_min = series[i].unwrap_or(0.0);
        let mut win_max = win_min;
        while j + 1 < n {
            let Some(next) = series[j + 1] else {
                break;
            };
            let new_min = win_min.min(next);
            let new_max = win_max.max(next);
            if new_max - new_min > max_log10_progress {
                break;
            }
            win_min = new_min;
            win_max = new_max;
            j += 1;
        }
        if j - i + 1 >= min_window {
            out.push(Stall {
                start_iter: iters[i].iter,
                end_iter: iters[j].iter,
                metric,
                delta_log10: win_max - win_min,
            });
            i = j + 1;
        } else {
            i += 1;
        }
    }
}

/// Contiguous runs of iters tagged `'r'` in the alpha-primal char
/// column — one entry per restoration entry → exit cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestorationWindow {
    pub start_iter: i32,
    pub end_iter: i32,
}

pub fn restoration_windows(report: &SolveReport) -> Vec<RestorationWindow> {
    let mut out: Vec<RestorationWindow> = Vec::new();
    let mut current: Option<RestorationWindow> = None;
    for r in &report.iterations {
        if r.alpha_primal_char.to_ascii_lowercase() == 'r' {
            match &mut current {
                Some(w) => w.end_iter = r.iter,
                None => {
                    current = Some(RestorationWindow {
                        start_iter: r.iter,
                        end_iter: r.iter,
                    })
                }
            }
        } else if let Some(w) = current.take() {
            out.push(w);
        }
    }
    if let Some(w) = current {
        out.push(w);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// One finding from [`diagnose`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub severity: Severity,
    /// Stable machine-readable identifier (e.g. `"max_iter_exceeded"`).
    pub code: &'static str,
    pub message: String,
}

/// Run common Ipopt-failure heuristics and return all findings.
///
/// Heuristics:
/// - `converged` (info): solver succeeded
/// - `max_iter_exceeded` (error): hit max_iter without converging
/// - `restoration_used` (warning): restoration phase entered ≥1 times
/// - `restoration_loop` (warning): multiple restoration entries
/// - `mu_stuck` (warning): barrier parameter barely decreased
/// - `heavy_line_search` (warning): backtracking ≥10 trials
/// - `hessian_regularized` (info): δ_w applied on any iter
/// - `convergence_stall` (warning): suppressed on clean convergence
///   unless the stall window is long (≥8 iters)
pub fn diagnose(report: &SolveReport) -> Vec<Finding> {
    let mut findings = Vec::new();
    let stats = &report.statistics;
    let solution = &report.solution;
    let iters = &report.iterations;
    let status = solution.status.as_str();

    if status == "SolveSucceeded" {
        findings.push(Finding {
            severity: Severity::Info,
            code: "converged",
            message: format!(
                "Solver converged in {} iterations to objective {:.6e}; KKT error {:.2e}.",
                stats.iteration_count, stats.final_objective, stats.final_kkt_error,
            ),
        });
    } else if status == "MaximumIterationsExceeded" {
        findings.push(Finding {
            severity: Severity::Error,
            code: "max_iter_exceeded",
            message: format!(
                "Hit max_iter without converging. KKT error at termination: {:.2e}. \
                 Consider raising max_iter, tightening initial guess, or relaxing tol.",
                stats.final_kkt_error,
            ),
        });
    }

    if stats.restoration_calls > 0 {
        findings.push(Finding {
            severity: Severity::Warning,
            code: "restoration_used",
            message: format!(
                "Restoration phase entered {} time(s); {} outer iters spent in \
                 restoration ({:.3}s). Indicates the line search couldn't make \
                 progress on the original problem.",
                stats.restoration_calls,
                stats.restoration_outer_iters,
                stats.restoration_wall_secs,
            ),
        });
    }

    if iters.len() >= 10 {
        let mu_first = iters[..3].iter().map(|r| r.mu).fold(0.0_f64, f64::max);
        let mu_last = iters[iters.len() - 3..]
            .iter()
            .map(|r| r.mu)
            .fold(f64::INFINITY, f64::min);
        if mu_first > 0.0 && mu_last > 0.0 {
            let log_drop = mu_first.log10() - mu_last.log10();
            if log_drop < 1.0 {
                findings.push(Finding {
                    severity: Severity::Warning,
                    code: "mu_stuck",
                    message: format!(
                        "Barrier parameter μ dropped only {log_drop:.2} orders of magnitude across \
                         {} iterations (from {mu_first:.2e} to {mu_last:.2e}). Try \
                         mu_strategy=adaptive or a smaller mu_init.",
                        iters.len(),
                    ),
                });
            }
        }
    }

    let heavy_ls: Vec<&IterRecord> = iters.iter().filter(|r| r.ls_trials >= 10).collect();
    if let Some(worst) = heavy_ls.iter().max_by_key(|r| r.ls_trials) {
        findings.push(Finding {
            severity: Severity::Warning,
            code: "heavy_line_search",
            message: format!(
                "{} iteration(s) needed >=10 backtracking trials (worst: iter {} with {} \
                 trials). Search direction quality may be poor — check Hessian accuracy.",
                heavy_ls.len(),
                worst.iter,
                worst.ls_trials,
            ),
        });
    }

    let big_reg: Vec<f64> = iters
        .iter()
        .map(|r| r.regularization)
        .filter(|&r| r > 1e-4)
        .collect();
    if !big_reg.is_empty() {
        let max_reg = big_reg.iter().copied().fold(0.0_f64, f64::max);
        findings.push(Finding {
            severity: Severity::Info,
            code: "hessian_regularized",
            message: format!(
                "Hessian regularization applied on {} iteration(s) (max δ_w = {max_reg:.2e}). \
                 The KKT system was indefinite; this is normal near saddle points but \
                 persistent regularization suggests a problematic Hessian.",
                big_reg.len(),
            ),
        });
    }

    let rwins = restoration_windows(report);
    if rwins.len() > 1 {
        findings.push(Finding {
            severity: Severity::Warning,
            code: "restoration_loop",
            message: format!(
                "Restoration was entered {} separate times. Repeated re-entry often means \
                 the problem is infeasible at the working point. Verify constraints.",
                rwins.len(),
            ),
        });
    }

    let stalls = find_stalls(report);
    if !stalls.is_empty() {
        let longest = stalls
            .iter()
            .map(|s| (s.end_iter - s.start_iter + 1) as usize)
            .max()
            .unwrap_or(0);
        if status != "SolveSucceeded" || longest >= 8 {
            findings.push(Finding {
                severity: Severity::Warning,
                code: "convergence_stall",
                message: format!(
                    "Detected {} stall window(s) where log-residual barely moved (longest: {} \
                     iters). Either the problem is ill-conditioned, scaling is off, or \
                     termination tolerance is too tight.",
                    stalls.len(),
                    longest,
                ),
            });
        }
    }

    findings
}

/// Augmented [`IterRecord`] returned by [`get_iterate`]: the raw row
/// plus derived log10 values handy for tooltip / LLM rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentedIterate {
    #[serde(flatten)]
    pub raw: IterRecord,
    pub log10_inf_pr: Option<f64>,
    pub log10_inf_du: Option<f64>,
    pub log10_mu: Option<f64>,
}

pub fn get_iterate(report: &SolveReport, k: usize) -> Result<AugmentedIterate, Error> {
    let n = report.iterations.len();
    if n == 0 {
        return Err(Error::NoIterations);
    }
    if k >= n {
        return Err(Error::IterOutOfRange { k, n });
    }
    let raw = report.iterations[k].clone();
    Ok(AugmentedIterate {
        log10_inf_pr: safe_log10(raw.inf_pr),
        log10_inf_du: safe_log10(raw.inf_du),
        log10_mu: safe_log10(raw.mu),
        raw,
    })
}

fn safe_log10(x: f64) -> Option<f64> {
    if x > 0.0 && x.is_finite() {
        Some(x.log10())
    } else {
        None
    }
}

/// One row in a side-by-side comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareRow {
    pub label: String,
    pub status: String,
    pub iter_count: i32,
    pub final_objective: f64,
    pub final_kkt_error: f64,
    pub restoration_calls: i32,
    pub elapsed_seconds: f64,
}

pub fn compare_runs<'a, I>(runs: I) -> Vec<CompareRow>
where
    I: IntoIterator<Item = (&'a str, &'a SolveReport)>,
{
    runs.into_iter()
        .map(|(label, r)| CompareRow {
            label: label.to_string(),
            status: r.solution.status.clone(),
            iter_count: r.statistics.iteration_count,
            final_objective: r.statistics.final_objective,
            final_kkt_error: r.statistics.final_kkt_error,
            restoration_calls: r.statistics.restoration_calls,
            elapsed_seconds: r.fair_metadata.elapsed_seconds,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::IterRecord;

    fn iter(idx: i32, mu: f64, inf_du: f64) -> IterRecord {
        IterRecord {
            iter: idx,
            inf_du,
            mu,
            alpha_primal_char: 'f',
            ..IterRecord::default()
        }
    }

    fn report_with(iters: Vec<IterRecord>) -> SolveReport {
        use crate::report::*;
        SolveReport {
            schema: SOLVE_REPORT_SCHEMA.into(),
            fair_metadata: FairMetadata {
                result_id: "t".into(),
                created_at_iso: "2026-05-24T00:00:00.000Z".into(),
                created_at_unix_nanos: 0,
                elapsed_seconds: 0.0,
                solver: SolverIdentity {
                    name: "pounce".into(),
                    version: "0.0.0".into(),
                    git_commit: None,
                    target_triple: "test".into(),
                },
                license: "EPL-2.0".into(),
                input: InputDescriptor::TnlpDirect,
            },
            problem: ProblemInfo {
                n_variables: 1,
                n_constraints: 0,
                n_objectives: 1,
                minimize: true,
                nnz_jac_g: None,
                nnz_h_lag: None,
            },
            solution: SolutionInfo {
                status: "SolveSucceeded".into(),
                solve_result_num: 0,
                objective: 0.0,
                x: vec![],
                lambda: vec![],
                suffixes: vec![],
            },
            statistics: StatisticsInfo {
                iteration_count: iters.len() as i32,
                final_objective: 0.0,
                final_scaled_objective: 0.0,
                final_dual_inf: 0.0,
                final_constr_viol: 0.0,
                final_compl: 0.0,
                final_kkt_error: 0.0,
                num_obj_evals: 0,
                num_constr_evals: 0,
                num_obj_grad_evals: 0,
                num_constr_jac_evals: 0,
                num_hess_evals: 0,
                total_wallclock_time_secs: 0.0,
                restoration_calls: 0,
                restoration_inner_iters: 0,
                restoration_outer_iters: 0,
                restoration_wall_secs: 0.0,
            },
            iterations: iters,
        }
    }

    #[test]
    fn stall_detection_flat_residual() {
        // 5 iters where inf_du barely moves -> one stall.
        let iters = (0..5)
            .map(|i| iter(i, 0.1, 1e-3 + (i as f64) * 1e-6))
            .collect();
        let stalls = find_stalls(&report_with(iters));
        assert_eq!(stalls.len(), 1);
        assert_eq!(stalls[0].start_iter, 0);
        assert_eq!(stalls[0].end_iter, 4);
    }

    #[test]
    fn stall_detection_progress_not_flagged() {
        // 5 iters where inf_du drops by orders of magnitude each step.
        let iters = (0..5)
            .map(|i| iter(i, 0.1, 10f64.powi(-i)))
            .collect();
        let stalls = find_stalls(&report_with(iters));
        assert!(stalls.is_empty(), "got {stalls:?}");
    }

    #[test]
    fn restoration_windows_grouped() {
        let mut iters = vec![iter(0, 0.1, 1e-2), iter(1, 0.1, 1e-3)];
        for i in 2..5 {
            let mut r = iter(i, 0.1, 1e-3);
            r.alpha_primal_char = 'r';
            iters.push(r);
        }
        iters.push(iter(5, 0.1, 1e-4));
        let windows = restoration_windows(&report_with(iters));
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_iter, 2);
        assert_eq!(windows[0].end_iter, 4);
    }

    #[test]
    fn get_iterate_out_of_range() {
        let report = report_with(vec![iter(0, 0.1, 1e-3)]);
        assert!(matches!(
            get_iterate(&report, 5),
            Err(Error::IterOutOfRange { k: 5, n: 1 }),
        ));
    }

    #[test]
    fn diagnose_clean_convergence_no_stall_warning() {
        // Quick converging run: just the `converged` finding, no stall noise.
        let iters: Vec<IterRecord> = (0..5)
            .map(|i| iter(i, 10f64.powi(-(i + 1)), 10f64.powi(-i)))
            .collect();
        let findings = diagnose(&report_with(iters));
        let codes: Vec<&str> = findings.iter().map(|f| f.code).collect();
        assert!(codes.contains(&"converged"), "got {codes:?}");
        assert!(
            !codes.contains(&"convergence_stall"),
            "stall shouldn't trip on healthy convergence: {codes:?}",
        );
    }
}
