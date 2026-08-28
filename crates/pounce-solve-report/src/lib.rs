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
    /// Solve-affecting environment variables present in the process
    /// environment at report time (`POUNCE_FERAL_*` numerics knobs and
    /// the legacy `FERAL_PIVTOL` / `FERAL_PARALLEL`). These alter the
    /// factorization or parallelism and can otherwise silently differ a
    /// run between two machines — e.g. one with `POUNCE_FERAL_PIVTOL`
    /// exported in a shell profile — with nothing in the report saying
    /// so (pounce#235). Recorded for reproducibility (the FAIR **R**
    /// principle). Presence here means the variable was set, not
    /// necessarily that it took effect: an explicit `OptionsList` setting
    /// (e.g. a `feral_pivtol` in an options file) takes precedence over
    /// the env fallback. Empty (and omitted from the JSON) when none are
    /// set. Additive — older `pounce.solve-report/v1` JSON without this
    /// field deserializes unchanged.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub environment: Vec<EnvOverride>,
}

/// One solve-affecting environment variable and its value, as captured
/// into [`FairMetadata::environment`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvOverride {
    /// Variable name, e.g. `POUNCE_FERAL_PIVTOL`.
    pub name: String,
    /// Its value verbatim, as read from the environment.
    pub value: String,
}

/// Environment variables that change pounce's numerics or parallelism —
/// the ones worth recording for reproducibility. Deliberately excludes the
/// `POUNCE_DBG_*` debug gates (they only add diagnostic output, never
/// altering the result) and compile-time / logging vars. Kept in a fixed
/// order so [`capture_solve_env_overrides`] is deterministic.
const SOLVE_AFFECTING_ENV_VARS: &[&str] = &[
    "POUNCE_FERAL_ORDERING",
    "POUNCE_FERAL_SCALING",
    "POUNCE_FERAL_PIVTOL",
    "POUNCE_FERAL_REFINE",
    "POUNCE_FERAL_CASCADE_BREAK",
    "POUNCE_FERAL_FMA",
    "POUNCE_FERAL_SINGULAR_PIVOT_FLOOR",
    "POUNCE_FERAL_MIN_PAR_FLOPS",
    // Legacy aliases without the `POUNCE_` prefix (still honored by the
    // FERAL backend): the deprecated pivot-threshold spelling and the
    // process-wide internal-parallelism switch.
    "FERAL_PIVTOL",
    "FERAL_PARALLEL",
];

/// Snapshot the solve-affecting environment variables ([`SOLVE_AFFECTING_ENV_VARS`])
/// that are currently set, for [`FairMetadata::environment`]. Reads the
/// process environment, so call it in the solving process. Returns them in
/// the fixed list order; absent variables are simply skipped.
pub fn capture_solve_env_overrides() -> Vec<EnvOverride> {
    SOLVE_AFFECTING_ENV_VARS
        .iter()
        .filter_map(|&name| {
            std::env::var(name).ok().map(|value| EnvOverride {
                name: name.to_string(),
                value,
            })
        })
        .collect()
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
    /// The same verdict in upstream Ipopt's C enumerator spelling —
    /// `Solve_Succeeded`, `Infeasible_Problem_Detected` — from
    /// `IpReturnCodes_inc.h`.
    ///
    /// [`Self::status`] carries the Rust variant name, which is *not* the
    /// name any Ipopt-facing consumer already keys off: CUTEst status
    /// tables, `benchmarks/scripts/run_nl_bench.sh`, the reference JSONs
    /// under `benchmarks/*/ipopt_ma57.json` and the CLI's own `Status:`
    /// line all spell it with separators. A consumer comparing
    /// `solution.status == "Solve_Succeeded"` against the report matched
    /// nothing and silently classified every solve as a failure (gh #767).
    /// This field is that spelling, so the comparison can be literal.
    ///
    /// Derived from [`Self::status`] by [`ReportBuilder::finish`] — never
    /// set by a caller, so the two cannot disagree. Empty when read back
    /// from a pre-#767 report.
    #[serde(default)]
    pub status_upstream: String,
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

/// NaN, for a residual slot that was never filled in.
fn uncomputed() -> Number {
    Number::NAN
}

/// Accept `null` for a residual the solve never computed.
///
/// `SolveStatistics` defaults its residual fields to NaN rather than `0.0`, so
/// that "the convergence check never ran" is distinguishable from "converged
/// exactly". `serde_json` renders a non-finite float as `null`, so a report
/// written for a solve that was refused during setup carries `null` in these
/// slots. Without this the report round-trip fails — pounce would write
/// reports its own `--cite` / studio / verify paths could not read back.
fn null_as_nan<'de, D>(de: D) -> Result<Number, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Number>::deserialize(de)?.unwrap_or_else(uncomputed))
}

/// Subset of `SolveStatistics` projected for the report. Mirrors the
/// fields the existing console summary prints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsInfo {
    pub iteration_count: Index,
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_objective: Number,
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_scaled_objective: Number,
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_dual_inf: Number,
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_constr_viol: Number,
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_compl: Number,
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_kkt_error: Number,
    /// The aggregate the strict convergence gate tested (gh #528): as
    /// `final_kkt_error`, but counting each constraint row's residual only
    /// where it exceeds what that row can represent in floating point. Equal
    /// to `final_kkt_error` unless a row is at its own resolution limit.
    #[serde(default = "uncomputed", deserialize_with = "null_as_nan")]
    pub final_kkt_error_above_noise: Number,
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
                // Overwritten from `status` by `finish`; see the field docs.
                status_upstream: String::new(),
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
            final_kkt_error_above_noise: src.final_kkt_error_above_noise,
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
        // Derived here rather than at each call site: every producer of a
        // report (CLI, C interface, Python bindings, the CBF driver) sets
        // `solution.status` and none of them can forget the upstream
        // spelling, nor set one that disagrees with the other (gh #767).
        let mut solution = self.solution;
        solution.status_upstream = solution.status.upstream_name().to_string();
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
                environment: capture_solve_env_overrides(),
            },
            problem: self.problem,
            solution,
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
        final_kkt_error_above_noise: 0.0,
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
///
/// `SolvedToAcceptableLevel` is `1`, not the 100 band, matching Ipopt's
/// ASL driver exactly (`Ipopt/src/Apps/AmplSolver/AmplTNLP.cpp`:
/// `STOP_AT_ACCEPTABLE_POINT` → `solve_result_num = 1`, message
/// "Solved To Acceptable Level."). The band is what consumers key on, and
/// the two bands are not interchangeable here: Pyomo's legacy `.sol`
/// reader turns `0..=99` into `status=ok` but `100..=199` into
/// `status=warning` with the same `termination_condition=optimal`, so the
/// 100 band made Pyomo log a "Loading a SolverResults object with a
/// warning status" warning on an accepted solve that Ipopt loads clean —
/// breaking solver-swappable clients whose accepted-solve contract
/// includes `status == ok` (gh #591). The reduced-accuracy convergence
/// stays visible in the status name and the `.sol` message line; it just
/// no longer reads as a warning.
///
/// `FeasiblePointFound` is `2`, Ipopt's own code, and therefore in the
/// `0..=99` solved band. It used to be `100`, justified by the claim that
/// the two statuses do not mean the same thing — that Ipopt returns
/// `FEASIBLE_POINT_FOUND` only for a square problem, where a feasible
/// point *is* the solution, while POUNCE used it more loosely for any
/// usable feasible point that missed the convergence criteria.
///
/// That claim was false about POUNCE's own code. The status has exactly
/// one production site: `min_c_1nrm.rs` returns
/// `RestorationOutcome::FeasiblePointFound`, reached only through the
/// gate at `resto_inner_solver.rs`, which is `is_square_problem && ...`.
/// `is_square_problem()` (`ipopt_alg.rs`) is `c.x.dim() == c.y_c.dim()`,
/// a port of `IpoptCalculatedQuantities::IsSquareProblem` — the same
/// condition Ipopt uses. So POUNCE emits this status *only* for square
/// problems, carrying Ipopt's meaning precisely, and on a square problem
/// there is no further convergence criterion to miss: the objective is
/// constant, so a feasible point is the solution.
///
/// The band is what consumers key on, and `100` was not a softer way of
/// saying the same thing — it inverted the answer. Pyomo's v2 reader
/// (`pyomo/contrib/solver/solvers/asl_sol_reader.py`) maps `100..=199` to
/// `TerminationCondition.error`, so a correct square-problem solve
/// reached the caller as a *solver error*; the legacy reader
/// (`pyomo/opt/plugins/sol.py`) maps it to `optimal` + `status=warning`,
/// the same gh #591 warning `SolvedToAcceptableLevel` was moved out of the
/// band to escape. Ipopt on the identical solve loads clean in both. This
/// is what gh #815 surfaced: an IDAES square flowsheet that POUNCE solves
/// to a constraint violation of 2.2e-06 was reported as a failure.
///
/// This crate is not the only place that had to agree. `python/pounce/gams/link.py`
/// already mapped the status to `(MODELSTAT_FEASIBLE, SOLVESTAT_NORMAL)`
/// and listed it as a success, and
/// `crates/pounce-algorithm/tests/issue_390_nonlinear_equality_scale.rs`
/// already called it "a success-band answer — AMPL `objno` code 2, which
/// every band table reads as SOLVED". The two Pyomo tables
/// (`pyomo_pounce.v2._V2_STATUS`, `pyomo_pounce.sens._STATUS_RESULT`) were
/// the dissenters and moved with this change.
///
/// Being in the solved band is *not* a claim that any feasible point is
/// acceptable. `issue_390_nonlinear_equality_scale.rs` is the guard that a
/// model with no solution is never reported feasible at any row scale; the
/// relative-violation threshold at `resto_inner_solver.rs` is what keeps
/// that true. This mapping decides how a verdict is reported, not when it
/// is reached.
pub fn status_to_solve_result_num(status: ApplicationReturnStatus) -> i32 {
    use ApplicationReturnStatus::*;
    match status {
        SolveSucceeded => 0,
        SolvedToAcceptableLevel => 1,
        FeasiblePointFound => 2,
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

    /// gh #767: the report is the FAIR-aligned machine surface, and a
    /// consumer keyed on Ipopt's own enumerator spelling — which is what
    /// CUTEst tables, the reference JSONs and the CLI's `Status:` line all
    /// use — must be able to compare a field literally. `status` carries
    /// the Rust variant name (`SolveSucceeded`); `status_upstream` carries
    /// `Solve_Succeeded`. A consumer that compared the former against the
    /// latter's spelling matched nothing and read every solve as a failure.
    #[test]
    fn report_carries_the_upstream_status_spelling_beside_the_rust_one() {
        let mut b = ReportBuilder::new(
            ReportDetail::Summary,
            InputDescriptor::Builtin {
                name: "rosenbrock".into(),
            },
        );
        b.solution.status = ApplicationReturnStatus::SolveSucceeded;
        let json = serde_json::to_value(b.finish()).expect("serialize");
        assert_eq!(json["solution"]["status"], "SolveSucceeded");
        assert_eq!(json["solution"]["status_upstream"], "Solve_Succeeded");
    }

    /// The derived field tracks whatever `status` was last set to — it is
    /// computed in `finish`, so a caller cannot leave it stale or set the
    /// two to different verdicts.
    #[test]
    fn upstream_status_spelling_is_derived_not_stored() {
        for status in [
            ApplicationReturnStatus::MaximumIterationsExceeded,
            ApplicationReturnStatus::InfeasibleProblemDetected,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
        ] {
            let mut b = ReportBuilder::new(
                ReportDetail::Summary,
                InputDescriptor::Builtin { name: "x".into() },
            );
            // Deliberately wrong; `finish` must overwrite it.
            b.solution.status_upstream = "Solve_Succeeded".to_string();
            b.solution.status = status;
            let report = b.finish();
            assert_eq!(report.solution.status_upstream, status.upstream_name());
        }
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

    /// gh #591: an accepted (reduced-accuracy) solve must land in AMPL's
    /// `0..=99` *solved* band with Ipopt's own code, `1`. In the 100 band
    /// Pyomo's legacy `.sol` reader loads the result as
    /// `status=warning, termination_condition=optimal` and logs a warning,
    /// while the identical Ipopt solve loads as `status=ok` — so a
    /// solver-swappable client that treats `status == ok` as part of its
    /// accepted-solve contract had to special-case POUNCE.
    #[test]
    fn solved_to_acceptable_level_is_in_the_solved_band_like_ipopt() {
        use ApplicationReturnStatus::*;
        let code = status_to_solve_result_num(SolvedToAcceptableLevel);
        assert_eq!(
            code, 1,
            "Ipopt's ASL driver emits 1 for STOP_AT_ACCEPTABLE_POINT",
        );
        assert!(
            (0..=99).contains(&code),
            "must be in the solved band Pyomo maps to status=ok, got {code}",
        );
        // Still distinguishable from a full-accuracy solve: the two codes
        // differ, and the status name carries the distinction verbatim into
        // the `.sol` message line.
        assert_ne!(code, status_to_solve_result_num(SolveSucceeded));

        // `FeasiblePointFound` is also in the solved band, for its own
        // reason — see `a_square_problem_feasible_point_is_in_the_solved_band`.
        assert_ne!(code, status_to_solve_result_num(FeasiblePointFound));
    }

    /// POUNCE emits `FeasiblePointFound` only for square problems — the
    /// status has one production site (`min_c_1nrm.rs`) behind one gate
    /// (`resto_inner_solver.rs`), and that gate is `is_square_problem &&
    /// ...`. That is exactly Ipopt's meaning, and on a square problem a
    /// feasible point *is* the solution, so the code is Ipopt's own `2`.
    ///
    /// The band, not the number, is what breaks: at `100` Pyomo's v2 ASL
    /// reader returns `TerminationCondition.error` for a correct solve
    /// (gh #815 — an IDAES flowsheet solved to a 2.2e-06 constraint
    /// violation and reported as a solver error), and the legacy reader
    /// returns `status=warning`, the same gh #591 complaint that moved
    /// `SolvedToAcceptableLevel` out of the band.
    #[test]
    fn a_square_problem_feasible_point_is_in_the_solved_band() {
        use ApplicationReturnStatus::*;
        let code = status_to_solve_result_num(FeasiblePointFound);
        assert_eq!(
            code, 2,
            "Ipopt's ASL driver emits 2 for FEASIBLE_POINT_FOUND"
        );
        assert!(
            (0..=99).contains(&code),
            "must be in the solved band both Pyomo readers accept, got {code}",
        );
        // Still its own verdict: distinguishable from both other members of
        // the band, with the distinction carried verbatim in the status name
        // and the `.sol` message line.
        assert_ne!(code, status_to_solve_result_num(SolveSucceeded));
        assert_ne!(code, status_to_solve_result_num(SolvedToAcceptableLevel));
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
