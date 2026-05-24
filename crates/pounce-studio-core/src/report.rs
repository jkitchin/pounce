//! `pounce.solve-report/v1` JSON types.
//!
//! Schema-compatible with the writer in
//! `crates/pounce-cli/src/solve_report.rs`. Re-defined here (rather than
//! imported) so this crate does not pull in the algorithm/CLI stack —
//! it should compile cleanly to `wasm32-unknown-unknown` for the
//! VS Code webview shell.
//!
//! Field-level drift between this definition and the writer is
//! caught by [`crate::tests`] / fixture round-trips: any new field
//! added to the writer either deserializes here (we use
//! `#[serde(default)]` and skip-empty where the writer does) or fails
//! the schema-tag check.

use serde::{Deserialize, Serialize};

/// JSON `schema` tag this crate understands. A report carrying any
/// other value is rejected by [`SolveReport::from_json_slice`].
pub const SOLVE_REPORT_SCHEMA: &str = "pounce.solve-report/v1";

#[derive(Debug)]
pub enum Error {
    /// JSON did not parse.
    Json(serde_json::Error),
    /// The top-level `schema` tag was not [`SOLVE_REPORT_SCHEMA`].
    SchemaMismatch { found: String },
    /// A requested iteration index was out of range.
    IterOutOfRange { k: usize, n: usize },
    /// The report carried no per-iteration history (writer ran at
    /// `--json-detail summary`).
    NoIterations,
    /// A binary iter-dump file was malformed (truncated, bad magic,
    /// unsupported version).
    IterDump(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid JSON: {e}"),
            Self::SchemaMismatch { found } => write!(
                f,
                "unexpected schema {found:?} (expected {SOLVE_REPORT_SCHEMA:?})",
            ),
            Self::IterOutOfRange { k, n } => {
                write!(f, "iter {k} out of range; report has {n} iterations")
            }
            Self::NoIterations => write!(
                f,
                "report has no iteration history (rerun with --json-detail full)",
            ),
            Self::IterDump(msg) => write!(f, "iter-dump parse error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveReport {
    pub schema: String,
    pub fair_metadata: FairMetadata,
    pub problem: ProblemInfo,
    pub solution: SolutionInfo,
    pub statistics: StatisticsInfo,
    #[serde(default)]
    pub iterations: Vec<IterRecord>,
}

impl SolveReport {
    /// Parse a JSON report from bytes. Validates the schema tag *first*
    /// (before full struct deserialization) so a mismatched version
    /// surfaces as [`Error::SchemaMismatch`] rather than a confusing
    /// "missing field" JSON error.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, Error> {
        #[derive(Deserialize)]
        struct SchemaProbe {
            schema: Option<String>,
        }
        let probe: SchemaProbe = serde_json::from_slice(bytes)?;
        let found = probe.schema.unwrap_or_default();
        if found != SOLVE_REPORT_SCHEMA {
            return Err(Error::SchemaMismatch { found });
        }
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Parse a JSON report from a `&str`. Convenience wrapper.
    pub fn from_json_str(s: &str) -> Result<Self, Error> {
        Self::from_json_slice(s.as_bytes())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairMetadata {
    pub result_id: String,
    pub created_at_iso: String,
    pub created_at_unix_nanos: i128,
    pub elapsed_seconds: f64,
    pub solver: SolverIdentity,
    pub license: String,
    pub input: InputDescriptor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverIdentity {
    pub name: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    pub target_triple: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum InputDescriptor {
    NlFile {
        path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        size_bytes: Option<u64>,
    },
    Builtin {
        name: String,
    },
    TnlpDirect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemInfo {
    pub n_variables: i32,
    pub n_constraints: i32,
    pub n_objectives: i32,
    pub minimize: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nnz_jac_g: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nnz_h_lag: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionInfo {
    /// Status string verbatim from
    /// `pounce_nlp::return_codes::ApplicationReturnStatus` — we keep it
    /// untyped here to avoid pulling in the nlp crate; consumers compare
    /// against known tags (`"SolveSucceeded"`, `"MaximumIterationsExceeded"`,
    /// etc.).
    pub status: String,
    pub solve_result_num: i32,
    pub objective: f64,
    #[serde(default)]
    pub x: Vec<f64>,
    #[serde(default)]
    pub lambda: Vec<f64>,
    #[serde(default)]
    pub suffixes: Vec<SolutionSuffix>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionSuffix {
    pub name: String,
    pub target: String,
    pub kind: String,
    #[serde(default)]
    pub values: Vec<f64>,
    #[serde(default)]
    pub int_values: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsInfo {
    pub iteration_count: i32,
    pub final_objective: f64,
    pub final_scaled_objective: f64,
    pub final_dual_inf: f64,
    pub final_constr_viol: f64,
    pub final_compl: f64,
    pub final_kkt_error: f64,
    pub num_obj_evals: i32,
    pub num_constr_evals: i32,
    pub num_obj_grad_evals: i32,
    pub num_constr_jac_evals: i32,
    pub num_hess_evals: i32,
    pub total_wallclock_time_secs: f64,
    pub restoration_calls: i32,
    pub restoration_inner_iters: i32,
    pub restoration_outer_iters: i32,
    pub restoration_wall_secs: f64,
}

/// One row of per-iteration trajectory; mirrors
/// `pounce_nlp::solve_statistics::IterRecord` field-by-field.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct IterRecord {
    pub iter: i32,
    pub objective: f64,
    pub inf_pr: f64,
    pub inf_du: f64,
    pub mu: f64,
    pub d_norm: f64,
    pub regularization: f64,
    pub alpha_dual: f64,
    pub alpha_primal: f64,
    /// Single-character tag (`f`, `h`, `r`, ...) describing the
    /// alpha-primal column. `'r'` indicates a restoration iteration.
    pub alpha_primal_char: char,
    pub ls_trials: i32,
}
