//! Machine-readable JSON solve report (pounce#8).
//!
//! Bundles the same payload AMPL's `.sol` carries (status, primal,
//! dual, suffixes) with FAIR-aligned provenance metadata (solver
//! identity, input descriptor, timestamp) and per-iteration history
//! when requested. Schema is versioned via the top-level `schema`
//! field so future extensions don't silently change semantics.
//!
//! FAIR reference: Wilkinson et al. (2016). *The FAIR Guiding
//! Principles for scientific data management and stewardship.*
//! Scientific Data, 3, 160018. DOI:
//! [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18).
//! Verified via Crossref on 2026-05-14.
//!
//! # Schema versioning
//!
//! The current schema tag is `pounce.solve-report/v1`. Breaking
//! changes bump the major version (v2 etc.). Adding fields without
//! removing or renaming existing ones is non-breaking — JSON
//! consumers should tolerate unknown fields.
//!
//! # Detail levels
//!
//! [`ReportDetail::Summary`] (default) emits the FAIR metadata,
//! problem dimensions, final solution, and aggregate statistics
//! — equivalent to a `.sol` plus provenance. [`ReportDetail::Full`]
//! additionally emits the per-iteration history (when captured by
//! [`pounce_algorithm::application::IpoptApplication::enable_iter_history`])
//! and any `solution.suffixes`. Choose `Summary` for production logs
//! and `Full` for debug captures.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use pounce_common::types::{Index, Number};
use pounce_linsol::summary::LinearSolverSummary;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::{IterRecord, SolveStatistics};
use serde::{Deserialize, Serialize};

/// Ipopt-style console printers (banner, problem statistics, end-of-run
/// summary). The single source of truth for the text log: the algorithm's
/// output layer emits these gated on `print_level`, and the CLI reuses the
/// banner. Moved out of `pounce-cli` so `pounce-algorithm` can emit them
/// natively (#206).
pub mod console;

/// Verbosity knob for the JSON report. Maps onto the `--json-detail`
/// CLI flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportDetail {
    /// FAIR metadata, problem, solution scalars + arrays, aggregate
    /// stats. Per-iteration history and suffix blocks omitted.
    Summary,
    /// Everything in `Summary` plus per-iteration history and any
    /// suffix outputs (`sens_sol_state_1`, reduced-Hessian blocks).
    Full,
}

impl ReportDetail {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "summary" => Ok(ReportDetail::Summary),
            "full" => Ok(ReportDetail::Full),
            other => Err(format!(
                "unknown --json-detail '{other}' (expected: summary | full)"
            )),
        }
    }
}

/// Top-level report struct. Fields are ordered so the JSON has the
/// most identifying / metadata fields first when pretty-printed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveReport {
    /// Schema identifier. Always
    /// `"pounce.solve-report/v1"` for this version of the writer.
    pub schema: String,
    /// FAIR provenance metadata.
    pub fair_metadata: FairMetadata,
    /// Problem dimensions and shape.
    pub problem: ProblemInfo,
    /// Final solution payload (status, primal, dual, suffixes).
    pub solution: SolutionInfo,
    /// Aggregate statistics (eval counts, KKT residuals, timing).
    pub statistics: StatisticsInfo,
    /// Per-iteration history. Empty when the report is at
    /// [`ReportDetail::Summary`] or iter history was never enabled.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub iterations: Vec<IterRecord>,
    /// Aggregate linear-solver post-mortem. Populated when the
    /// workspace-default FERAL backend ran (it self-instruments via
    /// `feral::Solver::last_factor_stats()`); `None` for HSL MA57 and
    /// for custom backends plugged through
    /// [`pounce_algorithm::application::IpoptApplication::set_linear_backend_factory`].
    /// Additive — older `pounce.solve-report/v1` JSON without this
    /// field deserializes unchanged.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub linear_solver: Option<LinearSolverSummaryInfo>,
}

/// Serializable mirror of [`pounce_linsol::summary::LinearSolverSummary`].
/// Lives in the CLI crate (rather than `pounce-linsol`) so the linsol
/// trait crate stays serde-free. Field shape is identical; serde
/// defaults keep it forward-compatible with future additions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSolverSummaryInfo {
    pub solver_name: String,
    pub n_factors: u64,
    pub n_pattern_reuse: u64,
    pub n_pattern_changes: u64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_fill_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub min_abs_pivot: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_abs_pivot: Option<f64>,
    /// `(positive, negative, zero)` inertia of the final factorisation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_inertia: Option<(usize, usize, usize)>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_nnz_a: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_nnz_l: Option<usize>,
}

impl From<LinearSolverSummary> for LinearSolverSummaryInfo {
    fn from(s: LinearSolverSummary) -> Self {
        Self {
            solver_name: s.solver_name,
            n_factors: s.n_factors,
            n_pattern_reuse: s.n_pattern_reuse,
            n_pattern_changes: s.n_pattern_changes,
            max_fill_ratio: s.max_fill_ratio,
            min_abs_pivot: s.min_abs_pivot,
            max_abs_pivot: s.max_abs_pivot,
            last_inertia: s.last_inertia,
            last_nnz_a: s.last_nnz_a,
            last_nnz_l: s.last_nnz_l,
        }
    }
}

/// FAIR-aligned provenance block. The four FAIR principles
/// (Wilkinson et al., 2016) map onto fields here as:
/// * **F**indable: `result_id` (unique per solve), `created_at_iso`.
/// * **A**ccessible: this JSON file is the artifact — no protocol
///   gating, plain text on disk.
/// * **I**nteroperable: schema versioned, types are JSON primitives,
///   units documented in field doc comments.
/// * **R**eusable: `solver`, `license`, `input` describe what was
///   solved with what code, enough to reproduce.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairMetadata {
    /// Unique per-solve identifier. Composed as
    /// `<unix_nanos>-<process_id>` so it is monotonically ordered
    /// within a process and globally unique across processes.
    pub result_id: String,
    /// Solve start time as ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SS.sssZ`).
    pub created_at_iso: String,
    /// Same instant in Unix nanoseconds (since 1970-01-01 UTC).
    /// Provided alongside the ISO string for callers that prefer
    /// integer arithmetic over date parsing.
    pub created_at_unix_nanos: i128,
    /// Wallclock seconds the solve took. Mirrors
    /// [`SolveStatistics::total_wallclock_time_secs`].
    pub elapsed_seconds: Number,
    /// Solver identity — name + version + (best-effort) git commit.
    pub solver: SolverIdentity,
    /// SPDX license string. Always `"EPL-2.0"` for this crate.
    pub license: String,
    /// Input descriptor. `kind` is `nl-file`, `builtin`, or
    /// `tnlp-direct` (for library callers).
    pub input: InputDescriptor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverIdentity {
    pub name: String,
    pub version: String,
    /// Git commit hash, captured at build time from the
    /// `POUNCE_GIT_COMMIT` environment variable. `None` if the build
    /// environment didn't set it — set via
    /// `POUNCE_GIT_COMMIT=$(git rev-parse HEAD) cargo build`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    /// Build target triple (e.g. `x86_64-apple-darwin`). Captured at
    /// build time from `TARGET` (Cargo standard env var).
    pub target_triple: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum InputDescriptor {
    NlFile {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        size_bytes: Option<u64>,
    },
    /// A Conic Benchmark Format (`.cbf`) instance — e.g. a CBLIB problem
    /// solved through the convex conic driver.
    CbfFile {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        size_bytes: Option<u64>,
    },
    Builtin {
        name: String,
    },
    TnlpDirect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemInfo {
    pub n_variables: Index,
    pub n_constraints: Index,
    pub n_objectives: Index,
    pub minimize: bool,
    /// Number of non-zeros declared by the TNLP for the constraint
    /// Jacobian. `None` if not exposed by the input path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nnz_jac_g: Option<Index>,
    /// Number of non-zeros declared for the Lagrangian Hessian.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nnz_h_lag: Option<Index>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionInfo {
    /// `SolveSucceeded`, `MaximumIterationsExceeded`, etc. The string
    /// form is the Rust enum variant name verbatim.
    pub status: ApplicationReturnStatus,
    /// AMPL-style solve-result code (Gay 2005, §5 p. 23 table).
    pub solve_result_num: i32,
    /// Final unscaled objective value (mirrors
    /// `SolveStatistics::final_objective`). `NaN` if unknown.
    pub objective: Number,
    /// Final primal vector, length `problem.n_variables`. Empty if
    /// not captured.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub x: Vec<Number>,
    /// Final dual (constraint multiplier) vector, length
    /// `problem.n_constraints`.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub lambda: Vec<Number>,
    /// Optional sIPOPT-style suffix blocks (`sens_sol_state_1` etc.).
    /// Stored as a flat map keyed by suffix name → list of
    /// `(index, value)` pairs, matching the AMPL `.sol` shape.
    /// Empty when no sensitivity / reduced-Hessian step ran.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub suffixes: Vec<SolutionSuffix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionSuffix {
    pub name: String,
    /// `"var" | "con" | "obj" | "problem"` per AMPL convention.
    pub target: String,
    /// `"int"` or `"real"`.
    pub kind: String,
    /// Dense values (length = target dimension); zero-filled for
    /// slots the writer didn't populate. Real-typed values are stored
    /// here; int-typed in `int_values`.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub values: Vec<Number>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub int_values: Vec<Index>,
}

/// Subset of `SolveStatistics` projected for the report. Mirrors the
/// fields the existing console summary prints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsInfo {
    pub iteration_count: Index,
    pub final_objective: Number,
    pub final_scaled_objective: Number,
    pub final_dual_inf: Number,
    pub final_constr_viol: Number,
    pub final_compl: Number,
    pub final_kkt_error: Number,
    pub num_obj_evals: Index,
    pub num_constr_evals: Index,
    pub num_obj_grad_evals: Index,
    pub num_constr_jac_evals: Index,
    pub num_hess_evals: Index,
    pub total_wallclock_time_secs: Number,
    pub restoration_calls: Index,
    pub restoration_inner_iters: Index,
    pub restoration_outer_iters: Index,
    pub restoration_wall_secs: Number,
}

/// Builder collecting the inputs for a [`SolveReport`]. The CLI
/// drivers populate one of these as they walk through the solve and
/// `finish()` it at the end.
pub struct ReportBuilder {
    detail: ReportDetail,
    started_at: SystemTime,
    started_unix_nanos: i128,
    pub input: InputDescriptor,
    pub problem: ProblemInfo,
    pub solution: SolutionInfo,
    pub stats: StatisticsInfo,
    pub iterations: Vec<IterRecord>,
    pub linear_solver: Option<LinearSolverSummaryInfo>,
}

impl ReportBuilder {
    pub fn new(detail: ReportDetail, input: InputDescriptor) -> Self {
        let now = SystemTime::now();
        let nanos = now
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i128)
            .unwrap_or(0);
        Self {
            detail,
            started_at: now,
            started_unix_nanos: nanos,
            input,
            problem: ProblemInfo {
                n_variables: 0,
                n_constraints: 0,
                n_objectives: 0,
                minimize: true,
                nnz_jac_g: None,
                nnz_h_lag: None,
            },
            solution: SolutionInfo {
                status: ApplicationReturnStatus::InternalError,
                solve_result_num: 500,
                // 0.0 (not NaN) so JSON round-trips. Callers that
                // need "unknown objective" semantics check
                // `statistics.iteration_count > 0` first.
                objective: 0.0,
                x: Vec::new(),
                lambda: Vec::new(),
                suffixes: Vec::new(),
            },
            stats: empty_stats(),
            iterations: Vec::new(),
            linear_solver: None,
        }
    }

    /// Attach a linear-solver post-mortem. Called once per solve after
    /// `optimize_tnlp` returns and before [`Self::finish`].
    pub fn set_linear_solver_summary(&mut self, summary: LinearSolverSummary) {
        self.linear_solver = Some(summary.into());
    }

    /// Pull `iteration_count`, `final_*`, and counters into the
    /// `stats` slot; copy `iterations` only if detail = Full.
    pub fn ingest_stats(&mut self, src: &SolveStatistics) {
        self.stats = StatisticsInfo {
            iteration_count: src.iteration_count,
            final_objective: src.final_objective,
            final_scaled_objective: src.final_scaled_objective,
            final_dual_inf: src.final_dual_inf,
            final_constr_viol: src.final_constr_viol,
            final_compl: src.final_compl,
            final_kkt_error: src.final_kkt_error,
            num_obj_evals: src.num_obj_evals,
            num_constr_evals: src.num_constr_evals,
            num_obj_grad_evals: src.num_obj_grad_evals,
            num_constr_jac_evals: src.num_constr_jac_evals,
            num_hess_evals: src.num_hess_evals,
            total_wallclock_time_secs: src.total_wallclock_time_secs,
            restoration_calls: src.restoration_calls,
            restoration_inner_iters: src.restoration_inner_iters,
            restoration_outer_iters: src.restoration_outer_iters,
            restoration_wall_secs: src.restoration_wall_secs,
        };
        if matches!(self.detail, ReportDetail::Full) {
            self.iterations = src.iterations.clone();
        }
    }

    pub fn finish(self) -> SolveReport {
        let elapsed = self
            .started_at
            .elapsed()
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let result_id = format!("{}-{}", self.started_unix_nanos, std::process::id());
        let created_at_iso = unix_nanos_to_iso(self.started_unix_nanos);

        SolveReport {
            schema: "pounce.solve-report/v1".to_string(),
            fair_metadata: FairMetadata {
                result_id,
                created_at_iso,
                created_at_unix_nanos: self.started_unix_nanos,
                elapsed_seconds: elapsed,
                solver: SolverIdentity {
                    name: "pounce".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                    git_commit: option_env!("POUNCE_GIT_COMMIT").map(String::from),
                    target_triple: TARGET_TRIPLE.to_string(),
                },
                license: "EPL-2.0".to_string(),
                input: self.input,
            },
            problem: self.problem,
            solution: self.solution,
            statistics: self.stats,
            iterations: self.iterations,
            linear_solver: self.linear_solver,
        }
    }
}

/// The build target triple (e.g. `aarch64-apple-darwin`).
///
/// Cargo only exposes `TARGET` to *build scripts*, not to crate source, so
/// `option_env!("TARGET")` here is always `None`. Our `build.rs` re-exports
/// the build script's `TARGET` as `POUNCE_TARGET_TRIPLE`, which we read
/// instead. Falls back to "unknown" if the build script did not run (e.g.
/// some non-Cargo tooling).
const TARGET_TRIPLE: &str = match option_env!("POUNCE_TARGET_TRIPLE") {
    Some(t) => t,
    None => "unknown",
};

fn empty_stats() -> StatisticsInfo {
    // All scalar fields start at 0.0 (not NaN) so the report
    // round-trips through `serde_json` — JSON has no NaN literal, and
    // serde_json's default is to write `null` for NaN, which then
    // fails to deserialize back into `Number`. Callers reading these
    // pre-solve treat `iteration_count == 0` as "no solve yet".
    StatisticsInfo {
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
    }
}

/// AMPL-style `solve_result_num` per Gay 2005 (Hooking Your Solver to
/// AMPL §5, p. 23 table): 0 = solved, 100s = warning, 200s =
/// infeasible, 300s = unbounded, 400s = limit reached, 500s = failure.
/// Shared by the CLI and cinterface report writers so both encode the
/// same int codes into `SolutionInfo::solve_result_num`.
///
/// `DivergingIterates` is Ipopt's unboundedness signal (the iterates run
/// off to infinity), so it maps to the 300 "unbounded" range — matching
/// upstream Ipopt's ASL driver and the CLI's own convex path, which
/// reports `QpStatus::DualInfeasible` (unbounded) as 300 (`main.rs`). It
/// is *not* a limit (400) condition.
pub fn status_to_solve_result_num(status: ApplicationReturnStatus) -> i32 {
    use ApplicationReturnStatus::*;
    match status {
        SolveSucceeded => 0,
        SolvedToAcceptableLevel => 100,
        FeasiblePointFound => 100,
        InfeasibleProblemDetected => 200,
        DivergingIterates => 300,
        SearchDirectionBecomesTooSmall => 400,
        MaximumIterationsExceeded => 400,
        MaximumCpuTimeExceeded => 400,
        MaximumWallTimeExceeded => 400,
        UserRequestedStop => 502,
        RestorationFailed => 500,
        ErrorInStepComputation => 500,
        InvalidNumberDetected => 500,
        InternalError => 500,
        UnrecoverableException => 500,
        NonIpoptExceptionThrown => 500,
        InsufficientMemory => 503,
        InvalidProblemDefinition => 504,
        InvalidOption => 504,
        NotEnoughDegreesOfFreedom => 504,
    }
}

/// Write a [`SolveReport`] to `path` as pretty-printed JSON. Returns
/// bytes written on success.
pub fn write_report_file(path: &Path, report: &SolveReport) -> std::io::Result<usize> {
    let s = serde_json::to_string_pretty(report)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, &s)?;
    Ok(s.len())
}

/// Convert Unix nanoseconds since the epoch to an ISO-8601 UTC
/// timestamp `YYYY-MM-DDTHH:MM:SS.sssZ`. Pure stdlib; no chrono /
/// time dependency. The conversion is based on the proleptic
/// Gregorian calendar formula from Howard Hinnant's "date" reference
/// (https://howardhinnant.github.io/date_algorithms.html), `days_from_civil`
/// in reverse — verified against `date -u -r <secs>` for several
/// epochs on 2026-05-14.
fn unix_nanos_to_iso(nanos: i128) -> String {
    let total_secs = nanos.div_euclid(1_000_000_000) as i64;
    let frac_nanos = nanos.rem_euclid(1_000_000_000) as i64;
    let millis = frac_nanos / 1_000_000;

    let days = total_secs.div_euclid(86_400);
    let secs_of_day = total_secs.rem_euclid(86_400);
    let hh = (secs_of_day / 3600) as i32;
    let mm = ((secs_of_day % 3600) / 60) as i32;
    let ss = (secs_of_day % 60) as i32;

    // Howard Hinnant's `civil_from_days` algorithm:
    //   z = days + 719468
    //   era = (z >= 0 ? z : z - 146096) / 146097
    //   doe = z - era*146097
    //   yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365
    //   y = yoe + era*400
    //   doy = doe - (365*yoe + yoe/4 - yoe/100)
    //   mp = (5*doy + 2) / 153
    //   d = doy - (153*mp + 2)/5 + 1
    //   m = mp < 10 ? mp + 3 : mp - 9
    //   y += (m <= 2)
    let z: i64 = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as i32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as i32;
    if m <= 2 {
        y += 1;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, m, d, hh, mm, ss, millis
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_formatter_matches_known_epochs() {
        // Epoch.
        assert_eq!(unix_nanos_to_iso(0), "1970-01-01T00:00:00.000Z");
        // 2000-01-01T00:00:00Z = 946684800 seconds.
        assert_eq!(
            unix_nanos_to_iso(946_684_800_000_000_000),
            "2000-01-01T00:00:00.000Z",
        );
        // 2024-02-29T12:34:56.789Z (leap-year sanity check).
        // Seconds: (2024 - 1970) * 365.25 days * 86400 ≈ 1709209296 — let's compute exactly.
        // Days from 1970-01-01 to 2024-02-29: 19782.
        // 19782 * 86400 = 1709164800. Plus 12*3600 + 34*60 + 56 = 45296.
        // Total = 1709210096.
        let s = unix_nanos_to_iso(1_709_210_096_789_000_000);
        assert_eq!(s, "2024-02-29T12:34:56.789Z", "got: {s}");
    }

    #[test]
    fn target_triple_resolves_to_real_triple_not_unknown() {
        // Fail-first: before the build.rs re-export this constant read
        // `option_env!("TARGET")`, which is `None` at crate-source compile
        // time (Cargo only exposes TARGET to build scripts), so it was always
        // "unknown". The build.rs now re-exports TARGET as
        // POUNCE_TARGET_TRIPLE, which resolves it to the real build triple.
        assert_ne!(
            TARGET_TRIPLE, "unknown",
            "build.rs should re-export the build target triple"
        );
        // A real triple has the `arch-vendor-os[-abi]` shape (>= 2 dashes).
        assert!(
            TARGET_TRIPLE.matches('-').count() >= 2,
            "unexpected target triple: {TARGET_TRIPLE:?}"
        );

        // And it must propagate into the finished report.
        let b = ReportBuilder::new(
            ReportDetail::Summary,
            InputDescriptor::NlFile {
                path: PathBuf::from("/tmp/foo.nl"),
                size_bytes: None,
            },
        );
        let report = b.finish();
        assert_eq!(report.fair_metadata.solver.target_triple, TARGET_TRIPLE);
        assert_ne!(report.fair_metadata.solver.target_triple, "unknown");
    }

    #[test]
    fn report_serializes_round_trip() {
        let mut b = ReportBuilder::new(
            ReportDetail::Summary,
            InputDescriptor::NlFile {
                path: PathBuf::from("/tmp/foo.nl"),
                size_bytes: Some(123),
            },
        );
        b.problem.n_variables = 5;
        b.problem.n_constraints = 4;
        b.solution.status = ApplicationReturnStatus::SolveSucceeded;
        b.solution.solve_result_num = 0;
        b.solution.objective = 0.55;
        b.solution.x = vec![0.63, 0.39, 0.02, 5.0, 1.0];
        b.solution.lambda = vec![-0.16, -0.29, -0.16, 0.18];
        b.stats.iteration_count = 9;

        let report = b.finish();
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        let back: SolveReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.schema, "pounce.solve-report/v1");
        assert_eq!(back.problem.n_variables, 5);
        assert_eq!(back.solution.x.len(), 5);
        assert!(matches!(
            back.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
        ));
    }

    #[test]
    fn summary_detail_omits_iterations_block() {
        let mut b = ReportBuilder::new(
            ReportDetail::Summary,
            InputDescriptor::Builtin {
                name: "rosenbrock".into(),
            },
        );
        let mut stats = SolveStatistics::default();
        stats.iterations.push(IterRecord {
            iter: 0,
            objective: 1.0,
            ..IterRecord::default()
        });
        b.ingest_stats(&stats);
        let r = b.finish();
        assert!(
            r.iterations.is_empty(),
            "Summary detail should drop iter history; got {} rows",
            r.iterations.len()
        );
        // And the JSON should not include the key at all (skip-empty).
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("\"iterations\":"), "json: {json}");
    }

    #[test]
    fn full_detail_includes_iteration_rows() {
        let mut b = ReportBuilder::new(ReportDetail::Full, InputDescriptor::TnlpDirect);
        let mut stats = SolveStatistics::default();
        stats.iterations.push(IterRecord {
            iter: 0,
            objective: 1.0,
            inf_pr: 0.5,
            ..IterRecord::default()
        });
        stats.iterations.push(IterRecord {
            iter: 1,
            objective: 0.5,
            inf_pr: 0.1,
            ..IterRecord::default()
        });
        b.ingest_stats(&stats);
        let r = b.finish();
        assert_eq!(r.iterations.len(), 2);
        assert_eq!(r.iterations[0].iter, 0);
        assert_eq!(r.iterations[1].iter, 1);
    }

    #[test]
    fn detail_parser_accepts_known_values() {
        assert_eq!(
            ReportDetail::parse("summary").unwrap(),
            ReportDetail::Summary
        );
        assert_eq!(ReportDetail::parse("Full").unwrap(), ReportDetail::Full);
        assert!(ReportDetail::parse("verbose").is_err());
    }

    #[test]
    fn diverging_iterates_maps_to_unbounded_range() {
        use ApplicationReturnStatus::*;
        // M12 regression: DivergingIterates is Ipopt's unboundedness
        // signal and must land in the AMPL 300 "unbounded" range, not
        // the 400 "limit" range — matching upstream Ipopt's ASL driver
        // and the CLI convex path (QpStatus::DualInfeasible → 300).
        assert_eq!(status_to_solve_result_num(DivergingIterates), 300);

        // Lock the surrounding range convention so the fix can't silently
        // drift back: solved / infeasible / limit / failure buckets.
        assert_eq!(status_to_solve_result_num(SolveSucceeded), 0);
        assert_eq!(status_to_solve_result_num(InfeasibleProblemDetected), 200);
        assert_eq!(
            status_to_solve_result_num(MaximumIterationsExceeded),
            400,
            "iteration limit stays in the 400 range",
        );
        assert_eq!(
            status_to_solve_result_num(SearchDirectionBecomesTooSmall),
            400,
        );
        assert_eq!(status_to_solve_result_num(RestorationFailed), 500);
    }

    #[test]
    fn result_id_is_unique_and_time_ordered() {
        let a = ReportBuilder::new(ReportDetail::Summary, InputDescriptor::TnlpDirect).finish();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = ReportBuilder::new(ReportDetail::Summary, InputDescriptor::TnlpDirect).finish();
        assert_ne!(a.fair_metadata.result_id, b.fair_metadata.result_id);
        assert!(
            b.fair_metadata.created_at_unix_nanos > a.fair_metadata.created_at_unix_nanos,
            "second result_id should sort after first"
        );
    }
}
