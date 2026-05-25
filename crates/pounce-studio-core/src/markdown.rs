//! Markdown renderer for solve-report summaries.
//!
//! Produces a one-screen Markdown document with the headline summary,
//! the convergence trajectory (sparkline-style ASCII bars), restoration
//! windows, and any [`Finding`]s from [`crate::analysis::diagnose`].
//! Used by the `pounce-studio inspect` CLI and intended as the free /
//! OSS deliverable from Phase 0.

use std::fmt::Write as _;

use crate::analysis::{
    convergence_trace, diagnose, find_stalls, restoration_windows, summarize, Severity,
};
use crate::report::SolveReport;

/// Render a complete Markdown inspection report.
pub fn render_inspect(report: &SolveReport) -> String {
    let mut s = String::new();
    let _ = render_inspect_into(report, &mut s);
    s
}

fn render_inspect_into(report: &SolveReport, out: &mut String) -> std::fmt::Result {
    let summary = summarize(report);
    let findings = diagnose(report);

    writeln!(out, "# Pounce solve report")?;
    writeln!(out)?;
    writeln!(out, "- **status**: `{}`", safe_code(&summary.status))?;
    writeln!(
        out,
        "- **solver**: {} {}",
        summary.solver, summary.solver_version
    )?;
    writeln!(out, "- **result_id**: `{}`", safe_code(&summary.result_id))?;
    writeln!(
        out,
        "- **problem**: {} vars, {} constraints",
        summary.n_variables, summary.n_constraints
    )?;
    writeln!(
        out,
        "- **iterations**: {} ({} captured in history)",
        summary.iteration_count, summary.iterations_captured
    )?;
    writeln!(
        out,
        "- **final objective**: {:.6e}",
        summary.final_objective
    )?;
    writeln!(out, "- **KKT error**: {:.2e}", summary.final_kkt_error)?;
    writeln!(
        out,
        "- **dual infeasibility**: {:.2e}",
        summary.final_dual_inf
    )?;
    writeln!(
        out,
        "- **constraint violation**: {:.2e}",
        summary.final_constr_viol
    )?;
    writeln!(out, "- **complementarity**: {:.2e}", summary.final_compl)?;
    writeln!(out, "- **elapsed**: {:.3} s", summary.elapsed_seconds)?;
    if summary.restoration_calls > 0 {
        writeln!(
            out,
            "- **restoration**: entered {} time(s), {} outer iters, {:.3} s",
            summary.restoration_calls,
            summary.restoration_outer_iters,
            summary.restoration_wall_secs,
        )?;
    }
    writeln!(out)?;

    if !findings.is_empty() {
        writeln!(out, "## Findings")?;
        writeln!(out)?;
        for f in &findings {
            let tag = match f.severity {
                Severity::Info => "info",
                Severity::Warning => "warning",
                Severity::Error => "error",
            };
            writeln!(
                out,
                "- **{tag}** `{}`: {}",
                safe_code(f.code),
                safe_text(&f.message),
            )?;
        }
        writeln!(out)?;
    }

    if !report.iterations.is_empty() {
        writeln!(out, "## Convergence trajectory")?;
        writeln!(out)?;
        writeln!(
            out,
            "| iter | objective | inf_pr | inf_du | mu | ‖d‖ | α_pr | ls |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|")?;
        let trace = convergence_trace(report);
        // Cap printed rows for readability; show first 20 and last 5
        // when N is large.
        let n = trace.iter.len();
        let to_show: Vec<usize> = if n <= 30 {
            (0..n).collect()
        } else {
            (0..20).chain((n - 5)..n).collect()
        };
        let mut last = None;
        for i in to_show {
            if let Some(prev) = last {
                if i != prev + 1 {
                    writeln!(out, "| ... | | | | | | | |")?;
                }
            }
            writeln!(
                out,
                "| {} | {:.4e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.3} | {} |",
                trace.iter[i],
                trace.objective[i],
                trace.inf_pr[i],
                trace.inf_du[i],
                trace.mu[i],
                trace.d_norm[i],
                trace.alpha_primal[i],
                trace.ls_trials[i],
            )?;
            last = Some(i);
        }
        writeln!(out)?;

        let stalls = find_stalls(report);
        if !stalls.is_empty() {
            writeln!(out, "## Stall windows")?;
            writeln!(out)?;
            writeln!(out, "| metric | iters | Δlog₁₀ |")?;
            writeln!(out, "|---|---|---|")?;
            for s in stalls {
                writeln!(
                    out,
                    "| {} | {}..{} | {:.3} |",
                    s.metric, s.start_iter, s.end_iter, s.delta_log10,
                )?;
            }
            writeln!(out)?;
        }

        let rwins = restoration_windows(report);
        if !rwins.is_empty() {
            writeln!(out, "## Restoration windows")?;
            writeln!(out)?;
            for w in rwins {
                writeln!(out, "- iters {}..{}", w.start_iter, w.end_iter)?;
            }
            writeln!(out)?;
        }
    }

    Ok(())
}

/// Sanitise a string for use inside a `` `...` `` inline code span.
///
/// CommonMark forbids escaping inside code spans — a backtick simply
/// terminates the span — so the only robust mitigation is to replace
/// the offending characters. We swap backticks for U+02CB (`ˋ`,
/// MODIFIER LETTER GRAVE ACCENT) which is visually similar and inert
/// in markdown.
fn safe_code(s: &str) -> String {
    if s.contains('`') {
        s.replace('`', "\u{02CB}")
    } else {
        s.to_string()
    }
}

/// Sanitise free text that ends up at the end of a list item or other
/// markdown context. Defends against breaking out of inline spans;
/// currently just neutralises stray backticks.
fn safe_text(s: &str) -> String {
    safe_code(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::*;

    fn minimal_report() -> SolveReport {
        SolveReport {
            schema: SOLVE_REPORT_SCHEMA.into(),
            fair_metadata: FairMetadata {
                result_id: "abc".into(),
                created_at_iso: "2026-05-24T00:00:00.000Z".into(),
                created_at_unix_nanos: 0,
                elapsed_seconds: 0.123,
                solver: SolverIdentity {
                    name: "pounce".into(),
                    version: "0.1.0".into(),
                    git_commit: None,
                    target_triple: "x86_64-unknown-linux-gnu".into(),
                },
                license: "EPL-2.0".into(),
                input: InputDescriptor::Builtin {
                    name: "test".into(),
                },
            },
            problem: ProblemInfo {
                n_variables: 2,
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
                iteration_count: 0,
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
            iterations: vec![],
            linear_solver: None,
        }
    }

    #[test]
    fn renders_headline_block() {
        let md = render_inspect(&minimal_report());
        assert!(md.contains("# Pounce solve report"));
        assert!(md.contains("SolveSucceeded"));
        assert!(md.contains("pounce 0.1.0"));
    }

    /// Backticks in code-span interpolations would terminate the span
    /// in CommonMark. `safe_code` replaces them with a modifier letter
    /// grave so the rendered markdown stays well-formed even when a
    /// status / result_id / finding code carries one.
    #[test]
    fn code_spans_escape_backticks() {
        let mut r = minimal_report();
        r.solution.status = "Weird`Status".into();
        r.fair_metadata.result_id = "id-with-`-tick".into();
        let md = render_inspect(&r);
        // No backtick should appear inside the interpolated values.
        let between = |md: &str, start: &str| {
            md.split(start)
                .nth(1)
                .and_then(|s| s.split('`').next())
                .unwrap_or("")
                .to_string()
        };
        assert!(!between(&md, "status**: `").contains('`'));
        assert!(!between(&md, "result_id**: `").contains('`'));
    }

    /// A report with no captured iterations should still render a
    /// valid Markdown document, just without the trajectory tables.
    #[test]
    fn renders_with_empty_iterations() {
        let md = render_inspect(&minimal_report());
        assert!(!md.contains("## Convergence trajectory"));
        assert!(!md.contains("## Stall windows"));
        assert!(!md.contains("## Restoration windows"));
    }
}
