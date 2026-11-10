//! `pounce diagnose <problem.nl>` — a model doctor that runs no solve.
//!
//! # Why this exists
//!
//! pounce's best model diagnostics used to be reachable only by solving:
//! the debugger's `diagnose` and `print rank` run inside `--debug`, Studio
//! reads a finished solve report, and the presolve structural checks surface
//! only as presolve warnings. A user holding an `Infeasible_Problem_Detected`
//! or `Invalid_Number_Detected` had no single command that asks "what is
//! wrong with this model?". This is that command (gh#1160, phases 1–2).
//!
//! # What it checks, in order
//!
//! | check | basis | source |
//! |---|---|---|
//! | `bounds` | proof | [`statics::check_bounds`] |
//! | `linear_rows` | proof | [`statics::check_linear_rows`] |
//! | `incidence` | proof | [`statics::check_incidence`] |
//! | `expression_domains`, `row_enclosures` | proof / heuristic | [`domain::check_domains`] |
//! | `coefficient_ranges` | proof | [`scaling::check_coefficients`] |
//! | `start_point` | numerical | `check-x0`, embedded as a section |
//! | `nonfinite_localization` | numerical | [`domain::check_nonfinite`] |
//! | `structural_rank` | structural | [`structure::check_structural`] |
//! | `numerical_rank` | numerical | [`structure::check_numerical`] |
//! | `jacobian_scaling` | numerical | [`scaling::check_jacobian`] |
//! | `rank_robustness` | numerical | [`robustness::check_robustness`] (phase 2) |
//! | `kkt_audit`, `licq`, `mfcq` | numerical | [`kkt::check_kkt`] (phase 2, needs `--sol`) |
//! | `hessian_inertia`, `reduced_hessian` | numerical | [`convexity::check_hessian`] (phase 2) |
//! | `convexity_sampling` | numerical | [`convexity::check_sampling`] (phase 2) |
//!
//! The numerical checks run at one point — the model's start, `--x0-file`,
//! or the primal values of a `.sol` (`--sol`) — and say so in every finding.
//! A finding about one point is never a claim about the model.
//!
//! The command never modifies the model and never solves it. Exit code `0`
//! when no finding is an error, `22` when one is, `2` on a usage or I/O
//! error. The schema is `pounce.diagnose/v1` (see [`report`]).
//!
//! User-facing documentation: `docs/src/diagnose.md`.

pub mod convexity;
pub mod domain;
pub mod kkt;
pub mod report;
pub mod robustness;
pub mod scaling;
pub mod statics;
pub mod structure;

#[cfg(test)]
mod phase2_tests;

use crate::check_x0::{
    CheckX0Args, CheckX0Outcome, check_tnlp_with_quadratics, clamp_to_interior, quad_row_coefs,
};
use crate::nl_reader::{self, NlProblem, NlTnlp};
use crate::verify::{name_at, sha256};
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};
use pounce_nlp::tnlp::{SparsityRequest, StartingPoint, TNLP};
use report::{Basis, Confidence, Finding, Report, Severity};
use serde_json::json;
use std::path::PathBuf;
use std::process::ExitCode;

/// The solver's default `bound_push` / `bound_frac`, for the clamp preview.
const BOUND_PUSH: Number = 1e-2;
const BOUND_FRAC: Number = 1e-2;

/// Tunables shared by the checks.
#[derive(Debug, Clone)]
pub struct DiagnoseOptions {
    /// Absolute/relative slack before a row or bound is called violated or
    /// empty (default 1e-6, the `check-x0` default).
    pub feas_tol: Number,
    /// Slack before an operator argument counts as outside its domain, to
    /// absorb outward rounding (default 1e-12).
    pub domain_tol: Number,
    /// Singular-value threshold relative to σ_max. Default
    /// `max(m, n)·ε`, the LAPACK/NumPy convention.
    pub rank_rel_tol: Option<Number>,
    /// Dense SVD limits: total entries and smaller dimension.
    pub max_dense_entries: usize,
    pub max_dense_dim: usize,
    /// How many minimal dependent sets to localize.
    pub max_dependent_sets: usize,
    /// Largest candidate set the deletion filter will shrink.
    pub max_dependent_search: usize,
    /// A magnitude ratio above this is a "wide" range (default 1e8).
    pub wide_range: Number,
    /// Findings kept per code (default 25).
    pub max_per_code: usize,
    /// Stationarity / zero-multiplier tolerance, relative to `max(1, ‖∇f‖∞)`
    /// (default 1e-6).
    pub opt_tol: Number,
    /// Inactive inequalities within this relative band of their bound are
    /// reported as near-active (default 1e-4).
    pub weak_band: Number,
    /// Seed for the convexity sampler (default 0).
    pub seed: u64,
    /// Sample pairs per function (default 64).
    pub convexity_samples: usize,
    /// Half-width of the sampling box, relative to `max(1, |x_j|)` (default 1).
    pub sample_radius: Number,
    /// Relative slack before a Jensen gap counts (default 1e-9).
    pub convexity_tol: Number,
}

impl Default for DiagnoseOptions {
    fn default() -> Self {
        DiagnoseOptions {
            feas_tol: 1e-6,
            domain_tol: 1e-12,
            rank_rel_tol: None,
            max_dense_entries: 4_000_000,
            max_dense_dim: 3_000,
            max_dependent_sets: 5,
            max_dependent_search: 200,
            wide_range: 1e8,
            max_per_code: 25,
            opt_tol: 1e-6,
            weak_band: 1e-4,
            seed: 0,
            convexity_samples: 64,
            sample_radius: 1.0,
            convexity_tol: 1e-9,
        }
    }
}

/// Parsed `diagnose` subcommand arguments.
#[derive(Debug, Clone, Default)]
pub struct DiagnoseArgs {
    pub nl: Option<PathBuf>,
    pub x0_file: Option<PathBuf>,
    pub sol: Option<PathBuf>,
    /// Replaces the point's label in the report (`--point-label`).
    pub point_label: Option<String>,
    /// Further points for `rank_robustness` (`--extra-point`, repeatable).
    pub extra_points: Vec<PathBuf>,
    pub json: bool,
    pub json_output: Option<PathBuf>,
    pub opts: DiagnoseOptions,
}

const USAGE: &str = "\
Usage: pounce diagnose <problem.nl> [OPTIONS]

Check a model without solving it. Reports, with the evidence for each:
contradictory bounds and rows (proved from the algebra), duplicate and
parallel rows, single-variable rows that are really bounds, variables that
appear nowhere, unbounded objective directions, operators whose arguments
the bounds do not keep inside their domain, rows that can never hold,
coefficient and scaling ranges, and - at one point - which operator made a
value or derivative NaN/Inf, structural and numerical rank of the equality
Jacobian, the smallest sets of dependent equations, and Jacobian scale
spread. The starting-point checks of `pounce check-x0` are included.

With --sol it also checks the solution: stationarity, multiplier signs and
complementarity, LICQ (with the dependent active constraints and how far
the multipliers can move), MFCQ, near-active and weakly active
constraints, and the reduced Hessian. At every point: whether a rank loss
survives other tolerances, scalings, a QR, and other points; the Hessian's
inertia; and sampled counterexamples to convexity.

Arguments:
  <problem.nl>              AMPL .nl problem

Point for the numerical checks (default: the model's starting point):
  --x0-file <path>          n whitespace-separated values
  --sol <path>              the primal values in an AMPL .sol file
  --point-label <text>      name the point in the report
  --extra-point <path>      a further point (n values) for rank_robustness;
                            repeatable

Options:
  --json                    print the JSON report (pounce.diagnose/v1)
  --json-output <path>      also write the JSON report to <path>
  --feas-tol <t>            slack for 'violated' / 'empty' (default 1e-6)
  --domain-tol <t>          slack for operator domains (default 1e-12)
  --rank-tol <t>            singular-value threshold relative to sigma_max
                            (default max(m,n)*eps)
  --max-dense-entries <k>   dense SVD limit on m*n (default 4000000)
  --max-dense-dim <k>       dense SVD limit on min(m,n) (default 3000)
  --max-dependent-sets <k>  minimal dependent sets to localize (default 5)
  --max-dependent-search <k>
                            largest set the minimal-set search shrinks
                            (default 200)
  --wide-range <r>          magnitude ratio reported as wide (default 1e8)
  --max-per-code <k>        findings kept per code (default 25)
  --opt-tol <t>             stationarity / zero-multiplier tolerance,
                            relative to max(1, |grad f|) (default 1e-6)
  --weak-band <t>           near-active band, relative (default 1e-4)
  --seed <k>                convexity sampler seed (default 0)
  --convexity-samples <k>   sample pairs per function (default 64)
  --sample-radius <r>       sampling box half-width, relative (default 1)
  --convexity-tol <t>       relative Jensen gap that counts as a
                            counterexample (default 1e-9)
  -h, --help                print this message

Exit code: 0 = no error findings (warnings allowed), 22 = at least one
error finding, 2 = usage/IO error.";

/// Entry point dispatched from `main` when argv[1] == "diagnose".
pub fn run_from_argv(rest: &[String]) -> ExitCode {
    let args = match parse_argv(rest) {
        Ok(Some(a)) => a,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("pounce diagnose: {msg}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    run(&args)
}

fn parse_argv(rest: &[String]) -> Result<Option<DiagnoseArgs>, String> {
    let mut a = DiagnoseArgs::default();
    let mut i = 0;
    fn value<'a>(rest: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
        *i += 1;
        rest.get(*i)
            .map(String::as_str)
            .ok_or_else(|| format!("{flag} needs a value"))
    }
    fn parse<T: std::str::FromStr>(s: &str, flag: &str) -> Result<T, String> {
        s.parse().map_err(|_| format!("{flag}: cannot parse `{s}`"))
    }
    while i < rest.len() {
        let arg = rest[i].as_str();
        match arg {
            "-h" | "--help" => return Ok(None),
            "--json" => a.json = true,
            "--json-output" => a.json_output = Some(value(rest, &mut i, arg)?.into()),
            "--x0-file" => a.x0_file = Some(value(rest, &mut i, arg)?.into()),
            "--sol" => a.sol = Some(value(rest, &mut i, arg)?.into()),
            "--point-label" => a.point_label = Some(value(rest, &mut i, arg)?.to_string()),
            "--feas-tol" => a.opts.feas_tol = parse(value(rest, &mut i, arg)?, arg)?,
            "--domain-tol" => a.opts.domain_tol = parse(value(rest, &mut i, arg)?, arg)?,
            "--rank-tol" => a.opts.rank_rel_tol = Some(parse(value(rest, &mut i, arg)?, arg)?),
            "--max-dense-entries" => {
                a.opts.max_dense_entries = parse(value(rest, &mut i, arg)?, arg)?
            }
            "--max-dense-dim" => a.opts.max_dense_dim = parse(value(rest, &mut i, arg)?, arg)?,
            "--max-dependent-sets" => {
                a.opts.max_dependent_sets = parse(value(rest, &mut i, arg)?, arg)?
            }
            "--max-dependent-search" => {
                a.opts.max_dependent_search = parse(value(rest, &mut i, arg)?, arg)?
            }
            "--wide-range" => a.opts.wide_range = parse(value(rest, &mut i, arg)?, arg)?,
            "--max-per-code" => a.opts.max_per_code = parse(value(rest, &mut i, arg)?, arg)?,
            "--extra-point" => a.extra_points.push(value(rest, &mut i, arg)?.into()),
            "--opt-tol" => a.opts.opt_tol = parse(value(rest, &mut i, arg)?, arg)?,
            "--weak-band" => a.opts.weak_band = parse(value(rest, &mut i, arg)?, arg)?,
            "--seed" => a.opts.seed = parse(value(rest, &mut i, arg)?, arg)?,
            "--convexity-samples" => {
                a.opts.convexity_samples = parse(value(rest, &mut i, arg)?, arg)?
            }
            "--sample-radius" => a.opts.sample_radius = parse(value(rest, &mut i, arg)?, arg)?,
            "--convexity-tol" => a.opts.convexity_tol = parse(value(rest, &mut i, arg)?, arg)?,
            s if s.starts_with('-') => return Err(format!("unknown option `{s}`")),
            s => {
                if a.nl.is_some() {
                    return Err(format!("unexpected extra argument `{s}`"));
                }
                a.nl = Some(s.into());
            }
        }
        i += 1;
    }
    if a.nl.is_none() {
        return Err("expected a <problem.nl> argument".into());
    }
    if a.x0_file.is_some() && a.sol.is_some() {
        return Err("--x0-file and --sol are mutually exclusive".into());
    }
    Ok(Some(a))
}

pub fn run(args: &DiagnoseArgs) -> ExitCode {
    let report = match diagnose_file(args) {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("pounce diagnose: {msg}");
            return ExitCode::from(2);
        }
    };
    let json_text =
        || serde_json::to_string_pretty(&report.to_json()).unwrap_or_else(|_| "{}".into());
    if args.json {
        println!("{}", json_text());
    } else {
        print!("{}", report.to_text());
    }
    if let Some(path) = &args.json_output {
        if let Err(e) = std::fs::write(path, json_text()) {
            eprintln!(
                "pounce diagnose: failed to write report {}: {e}",
                path.display()
            );
            return ExitCode::from(2);
        }
        if !args.json {
            println!("  report: {}", path.display());
        }
    }
    if report.has_errors() {
        ExitCode::from(22)
    } else {
        ExitCode::SUCCESS
    }
}

/// Load the `.nl`, pick the point, and run every check.
pub fn diagnose_file(args: &DiagnoseArgs) -> Result<Report, String> {
    let path = args.nl.as_ref().ok_or("expected a <problem.nl> argument")?;
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let prob = nl_reader::read_nl_file(path)?;
    if !prob.complementarity.is_empty() {
        return Err(format!(
            "{} declares {} complementarity pair(s), which diagnose does not model",
            path.display(),
            prob.complementarity.len()
        ));
    }
    let read_values = |p: &PathBuf| -> Result<Vec<Number>, String> {
        let text =
            std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
        let vals: Result<Vec<Number>, _> = text.split_whitespace().map(str::parse).collect();
        vals.map_err(|e| format!("{}: bad value: {e}", p.display()))
    };
    let mut inputs = Inputs::default();
    if let Some(p) = &args.sol {
        let text =
            std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
        let sol = crate::verify::parse_sol(&text)?;
        inputs.point = Some((format!("the solution in {}", p.display()), sol.x));
        inputs.multipliers = (!sol.lambda.is_empty()).then_some(sol.lambda);
        if sol.z_l.is_some() || sol.z_u.is_some() {
            let n = prob.n;
            inputs.bound_multipliers = Some((
                sol.z_l.unwrap_or_else(|| vec![0.0; n]),
                sol.z_u.unwrap_or_else(|| vec![0.0; n]),
            ));
        }
        inputs.is_solution = true;
    } else if let Some(p) = &args.x0_file {
        inputs.point = Some((format!("the point in {}", p.display()), read_values(p)?));
    }
    if let (Some(pt), Some(l)) = (inputs.point.as_mut(), &args.point_label) {
        pt.0 = l.clone();
    }
    for p in &args.extra_points {
        inputs
            .extra_points
            .push((format!("the point in {}", p.display()), read_values(p)?));
    }
    let mut report = diagnose_problem(&prob, &inputs, path.display().to_string(), &args.opts)?;
    report.sha256 = Some(sha256::hex(&bytes));
    Ok(report)
}

/// What to evaluate at, beyond the model itself.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    /// Overrides the model's starting point: (label, values).
    pub point: Option<(String, Vec<Number>)>,
    /// Constraint multipliers at `point`, as a `.sol` carries them.
    pub multipliers: Option<Vec<Number>>,
    /// Bound multipliers `(z_L, z_U)` at `point`, from the `.sol`'s
    /// `ipopt_zL_out` / `ipopt_zU_out` suffixes when present.
    pub bound_multipliers: Option<(Vec<Number>, Vec<Number>)>,
    /// `point` is where a solve stopped (enables the KKT checks).
    pub is_solution: bool,
    /// Further points for `rank_robustness`.
    pub extra_points: Vec<(String, Vec<Number>)>,
}

/// Run every check on an in-memory problem.
pub fn diagnose_problem(
    prob: &NlProblem,
    inputs: &Inputs,
    source: String,
    opts: &DiagnoseOptions,
) -> Result<Report, String> {
    let point = inputs.point.clone();
    let mut report = Report::new(source.clone(), prob.n, prob.m, opts.max_per_code);
    report.tolerances.insert("feas_tol", opts.feas_tol);
    report.tolerances.insert("domain_tol", opts.domain_tol);
    report.tolerances.insert("wide_range", opts.wide_range);
    report
        .tolerances
        .insert("convexity_tol", opts.convexity_tol);

    // Static checks: the model as written.
    statics::check_bounds(prob, &mut report);
    statics::check_linear_rows(prob, opts, &mut report);
    statics::check_incidence(prob, &mut report);
    domain::check_domains(prob, opts, &mut report);
    let quad = quad_row_coefs(prob);
    scaling::check_coefficients(prob, &quad, opts, &mut report);

    // Point checks.
    let mut tnlp = NlTnlp::try_new(prob.clone())?;
    let x = match &point {
        Some((label, v)) => {
            if v.len() != prob.n {
                return Err(format!(
                    "{label} has {} values but the problem has {} variables",
                    v.len(),
                    prob.n
                ));
            }
            v.clone()
        }
        None => starting_point(&mut tnlp, prob.n)?,
    };
    let label = point
        .as_ref()
        .map(|p| p.0.clone())
        .unwrap_or_else(|| "the model's starting point".to_string());
    report.point = label.clone();

    let x0_args = CheckX0Args {
        x0_values: Some((label.clone(), x.clone())),
        feas_tol: opts.feas_tol,
        ..Default::default()
    };
    // A solve starts from a point only after its interior clamp; if the
    // clamp moves it, evaluate there too so a non-finite value the clamp
    // removes is not reported as fatal. A solution is a point a solve
    // already reached — often exactly on an active bound — so it is never
    // clamped: what is non-finite there is non-finite.
    let clamped: Vec<Number> = (0..prob.n)
        .map(|j| clamp_to_interior(x[j], prob.x_l[j], prob.x_u[j], BOUND_PUSH, BOUND_FRAC))
        .collect();
    let clamped = (!inputs.is_solution && clamped != x).then_some(clamped);
    let hess_after_clamp = clamped
        .as_ref()
        .and_then(|c| hessian_nonfinite(&mut tnlp, c, prob.m));
    match check_tnlp_with_quadratics(
        &mut tnlp,
        &prob.var_names,
        &prob.con_names,
        None,
        source,
        &quad,
        &x0_args,
    ) {
        Ok(o) => start_point_findings(
            &o,
            &label,
            inputs.is_solution,
            hess_after_clamp,
            &mut report,
        ),
        Err(e) => report.unavailable("start_point", e),
    }
    let clamped_pt = clamped
        .map(|c| PointEval::evaluate(&mut tnlp, c, format!("{label} after the interior clamp")));
    let mut pt = PointEval::evaluate(&mut tnlp, x, label);
    pt.is_solution = inputs.is_solution;
    domain::check_nonfinite(prob, &pt, clamped_pt.as_ref(), &mut report);
    let sr = structure::check_structural(prob, &pt, &mut report);
    structure::check_numerical(prob, &pt, sr, opts, &mut report);
    scaling::check_jacobian(prob, &pt, opts, &mut report);

    // Phase 2. Further points for the robustness check: the model's start
    // (when the main point is another one) and each --extra-point.
    let mut others: Vec<PointEval> = Vec::new();
    if point.is_some() {
        let x0 = starting_point(&mut tnlp, prob.n)?;
        others.push(PointEval::evaluate(
            &mut tnlp,
            x0,
            "the model's starting point".into(),
        ));
    }
    for (l, v) in &inputs.extra_points {
        if v.len() != prob.n {
            return Err(format!(
                "{l} has {} values but the problem has {} variables",
                v.len(),
                prob.n
            ));
        }
        others.push(PointEval::evaluate(&mut tnlp, v.clone(), l.clone()));
    }
    robustness::check_robustness(prob, &pt, &others, opts, &mut report);
    let kkt = kkt::check_kkt(
        prob,
        &pt,
        inputs.multipliers.as_deref(),
        inputs
            .bound_multipliers
            .as_ref()
            .map(|(l, u)| (l.as_slice(), u.as_slice())),
        opts,
        &mut report,
    );
    convexity::check_hessian(prob, &mut tnlp, &pt, kkt.as_ref(), opts, &mut report);
    // Every check either ran or says why not: reduced_hessian is skipped
    // without a solution, and also whenever hessian_inertia returned early.
    if !report.checks.iter().any(|c| c.name == "reduced_hessian") {
        report.unavailable(
            "reduced_hessian",
            if kkt.is_none() {
                "needs a solution with multipliers: pass the solve's .sol with --sol"
            } else {
                "the Hessian could not be evaluated here (see hessian_inertia)"
            },
        );
    }
    convexity::check_sampling(prob, &pt, opts, &mut report);
    Ok(report)
}

fn starting_point(tnlp: &mut dyn TNLP, n: usize) -> Result<Vec<Number>, String> {
    let m = tnlp
        .get_nlp_info()
        .map(|i| i.m.max(0) as usize)
        .unwrap_or(0);
    let mut x = vec![0.0; n];
    let (mut zl, mut zu, mut lam) = (vec![0.0; n], vec![0.0; n], vec![0.0; m]);
    if !tnlp.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut zl,
        z_u: &mut zu,
        init_lambda: false,
        lambda: &mut lam,
    }) {
        return Err("get_starting_point failed".into());
    }
    Ok(x)
}

/// The `check-x0` outcome as findings, plus the full report as a section.
/// Non-finite entries in the Hessian of the Lagrangian at `x` (objective
/// factor 1, zero multipliers, as `check-x0` evaluates it). `None` when the
/// evaluator has no exact Hessian.
fn hessian_nonfinite(tnlp: &mut dyn TNLP, x: &[Number], m: usize) -> Option<usize> {
    let nnz = tnlp.get_nlp_info()?.nnz_h_lag.max(0) as usize;
    if nnz == 0 {
        return Some(0);
    }
    let (mut hr, mut hc, mut hv) = (vec![0i32; nnz], vec![0i32; nnz], vec![0.0; nnz]);
    let lambda = vec![0.0; m];
    let ok = tnlp.eval_h(
        None,
        false,
        1.0,
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut hr,
            jcol: &mut hc,
        },
    ) && tnlp.eval_h(
        Some(x),
        true,
        1.0,
        Some(&lambda),
        true,
        SparsityRequest::Values { values: &mut hv },
    );
    ok.then(|| hv.iter().filter(|v| !v.is_finite()).count())
}

/// `is_solution`: the point is a `.sol` a solve reached, so start-point
/// advice (the clamp, an all-zero start) does not apply. `hess_after_clamp`:
/// the Hessian's non-finite count at the clamped start, when the clamp
/// moved it.
fn start_point_findings(
    o: &CheckX0Outcome,
    label: &str,
    is_solution: bool,
    hess_after_clamp: Option<usize>,
    report: &mut Report,
) {
    report
        .sections
        .insert("check_x0", crate::check_x0::report_value(o));
    let mut sink = report.ran("start_point");
    let rows =
        |v: &[crate::verify::RowReport]| v.iter().map(|r| r.name.clone()).collect::<Vec<_>>();
    if o.n_bound_violations > 0 {
        sink.push(
            Finding::new("start_point", "start_violates_variable_bounds", Severity::Warning,
                Basis::Numerical, Confidence::Certain,
                format!("{} variable(s) of {label} lie outside their bounds (worst by {:.3e}).", o.n_bound_violations, o.max_bound_violation))
            .why("The solver projects the start into the box before iteration 0, so the point you supplied is not the point it starts from.")
            .next("Check how the start was built (units, a stale warm start, an index shift).")
            .vars(rows(&o.bound_violations))
            .evidence(json!({"count": o.n_bound_violations, "max_violation": o.max_bound_violation})),
        );
    }
    if o.x0_all_zero && o.n_vars > 0 && !is_solution {
        sink.push(
            Finding::new("start_point", "start_all_zero", Severity::Info, Basis::Numerical,
                Confidence::Certain,
                format!("Every component of {label} is zero."))
            .why("Usually means no starting point was supplied; zero is often a poor or singular start for a nonlinear model.")
            .next("Supply a starting point from the model's meaning, or check one with --x0-file."),
        );
    }
    if o.n_clamp_moved > 0 && !is_solution {
        sink.push(
            Finding::new("start_point", "start_moved_by_bound_push", Severity::Info, Basis::Numerical,
                Confidence::Certain,
                format!("The interior clamp (bound_push) moves {} component(s) of {label}, by up to {:.3e}.", o.n_clamp_moved, o.max_clamp_move))
            .why("Components on or too near a bound are pushed inside; a warm start loses exactly those components.")
            .next("See sections.check_x0.interior_clamp for which components move.")
            .evidence(json!({"count": o.n_clamp_moved, "max_move": o.max_clamp_move})),
        );
    }
    if o.n_con_violations > 0 {
        sink.push(
            Finding::new("start_point", "start_constraint_violation",
                if is_solution { Severity::Warning } else { Severity::Info }, Basis::Numerical,
                Confidence::Certain,
                format!("{} constraint(s) are violated at {label}; the largest violation is {:.3e}.", o.n_con_violations, o.max_con_violation))
            .why(if is_solution {
                "A solution that violates its constraints beyond the tolerance is not feasible; check the solve's status."
            } else {
                "Infeasible starts are normal for an interior-point method, but a very large violation usually means a wrong or missing starting point."
            })
            .next(if is_solution {
                "Compare with `pounce verify`, and the worst rows in sections.check_x0."
            } else {
                "Compare the worst rows (sections.check_x0) with what the start was meant to satisfy."
            })
            .cons(rows(&o.con_violations))
            .evidence(json!({"count": o.n_con_violations, "max_violation": o.max_con_violation})),
        );
    }
    if let Some(h) = o.hess_nonfinite_count
        && h > 0
    {
        // Downgrade only when the Hessian was re-evaluated at the clamped
        // start and is finite there.
        let cleared = hess_after_clamp == Some(0);
        sink.push(
                Finding::new("start_point", "nonfinite_hessian_at_point",
                    if cleared { Severity::Warning } else { Severity::Error },
                    Basis::Numerical, Confidence::Certain,
                    format!("The Hessian of the Lagrangian has {h} non-finite entr{} at {label}.", if h == 1 { "y" } else { "ies" }))
                .why(if cleared {
                    "The interior clamp moves this start, and the Hessian is finite at the clamped point, so a default solve does not see it; a solve that evaluates the Hessian here aborts with Invalid_Number_Detected."
                } else {
                    "A solve that needs the Hessian here aborts with Invalid_Number_Detected."
                })
                .next("Look for second-derivative poles (sqrt, log, fractional powers near 0) in the nonlinear rows.")
                .evidence(json!({"count": h, "count_after_interior_clamp": hess_after_clamp})),
            );
    }
}

/// Values and first derivatives at one point, from the solver's own
/// evaluator (the same callbacks a solve would call).
#[derive(Debug, Clone)]
pub struct PointEval {
    pub label: String,
    pub x: Vec<Number>,
    pub f: Option<Number>,
    pub grad: Vec<Number>,
    pub grad_ok: bool,
    pub g: Vec<Number>,
    pub g_ok: bool,
    /// Jacobian triplets, 0-based.
    pub jac_irow: Vec<usize>,
    pub jac_jcol: Vec<usize>,
    pub jac_val: Vec<Number>,
    pub jac_structure_ok: bool,
    pub jac_ok: bool,
    /// The point is where a solve stopped (`--sol`).
    pub is_solution: bool,
}

impl PointEval {
    pub fn evaluate(tnlp: &mut dyn TNLP, x: Vec<Number>, label: String) -> Self {
        let info = tnlp.get_nlp_info();
        let (n, m, nnz, off) = match &info {
            Some(i) => (
                i.n.max(0) as usize,
                i.m.max(0) as usize,
                i.nnz_jac_g.max(0) as usize,
                usize::from(matches!(
                    i.index_style,
                    pounce_nlp::tnlp::IndexStyle::Fortran
                )),
            ),
            None => (x.len(), 0, 0, 0),
        };
        let f = tnlp.eval_f(&x, true);
        let mut grad = vec![0.0; n];
        let grad_ok = tnlp.eval_grad_f(&x, false, &mut grad);
        let mut g = vec![0.0; m];
        let g_ok = m == 0 || tnlp.eval_g(&x, false, &mut g);
        let mut irow = vec![0i32; nnz];
        let mut jcol = vec![0i32; nnz];
        let mut jval = vec![0.0; nnz];
        let jac_structure_ok = nnz == 0
            || tnlp.eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut irow,
                    jcol: &mut jcol,
                },
            );
        let jac_ok = jac_structure_ok
            && (nnz == 0
                || tnlp.eval_jac_g(
                    Some(&x),
                    false,
                    SparsityRequest::Values { values: &mut jval },
                ));
        PointEval {
            label,
            x,
            f,
            grad,
            grad_ok,
            g,
            g_ok,
            jac_irow: irow
                .iter()
                .map(|&r| (r as usize).wrapping_sub(off))
                .collect(),
            jac_jcol: jcol
                .iter()
                .map(|&c| (c as usize).wrapping_sub(off))
                .collect(),
            jac_val: jval,
            jac_structure_ok,
            jac_ok,
            is_solution: false,
        }
    }

    /// Evaluate an in-memory problem (builds its evaluator). Test helper.
    #[cfg(test)]
    pub(crate) fn evaluate_nl(prob: &NlProblem, x: Vec<Number>, label: String) -> Self {
        let mut t = NlTnlp::try_new(prob.clone()).expect("evaluator");
        Self::evaluate(&mut t, x, label)
    }
}

pub(crate) fn vname(prob: &NlProblem, j: usize) -> String {
    name_at(&prob.var_names, j, 'x')
}

pub(crate) fn cname(prob: &NlProblem, i: usize) -> String {
    name_at(&prob.con_names, i, 'c')
}

/// A lower bound as an interval endpoint (`-∞` when absent).
pub(crate) fn lo_eff(b: Number) -> Number {
    if lower_bound_present(b) {
        b
    } else {
        Number::NEG_INFINITY
    }
}

/// An upper bound as an interval endpoint (`+∞` when absent).
pub(crate) fn up_eff(b: Number) -> Number {
    if upper_bound_present(b) {
        b
    } else {
        Number::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, NlProblemParts, UnaryOp};

    #[test]
    fn healthy_problem_has_no_errors_or_warnings() {
        // min (x0-1)^2 + (x1-2)^2  s.t. x0 + x1 = 3, x0*x1 >= 1, x in [0, 10].
        let sq = |j: usize, c: Number| {
            Expr::Binary(
                BinOp::Pow,
                Box::new(Expr::Binary(
                    BinOp::Sub,
                    Box::new(Expr::Var(j)),
                    Box::new(Expr::Const(c)),
                )),
                Box::new(Expr::Const(2.0)),
            )
        };
        let p = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Binary(BinOp::Add, Box::new(sq(0, 1.0)), Box::new(sq(1, 2.0))),
            obj_constant: 0.0,
            constraints: vec![
                Expr::Binary(BinOp::Add, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
            ],
            x_l: vec![0.0, 0.0],
            x_u: vec![10.0, 10.0],
            x0: vec![1.5, 1.5],
            g_l: vec![3.0, 1.0],
            g_u: vec![3.0, 1e19],
            var_names: vec!["a".into(), "b".into()],
            con_names: vec!["sum".into(), "prod".into()],
        })
        .unwrap();
        let r = diagnose_problem(
            &p,
            &Inputs::default(),
            "t".into(),
            &DiagnoseOptions::default(),
        )
        .unwrap();
        let bad: Vec<_> = r
            .findings
            .iter()
            .filter(|f| f.severity != Severity::Info)
            .map(|f| (f.code, f.observation.clone()))
            .collect();
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(r.verdict(), "CLEAN");
        // Every check either ran or said why not.
        let names: Vec<_> = r.checks.iter().map(|c| c.name).collect();
        for c in [
            "bounds",
            "linear_rows",
            "incidence",
            "expression_domains",
            "row_enclosures",
            "coefficient_ranges",
            "start_point",
            "nonfinite_localization",
            "structural_rank",
            "numerical_rank",
            "jacobian_scaling",
        ] {
            assert!(names.contains(&c), "missing check {c}: {names:?}");
        }
        let j = r.to_json();
        assert_eq!(j["schema"], "pounce.diagnose/v1");
        assert_eq!(j["verdict"], "CLEAN");
    }

    #[test]
    fn nan_start_is_an_error_and_names_the_operator() {
        let p = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Var(0),
            obj_constant: 0.0,
            constraints: vec![Expr::Unary(UnaryOp::Log, Box::new(Expr::Var(0)))],
            x_l: vec![-1e19],
            x_u: vec![1e19],
            x0: vec![-1.0],
            g_l: vec![0.0],
            g_u: vec![1e19],
            var_names: vec![],
            con_names: vec![],
        })
        .unwrap();
        let r = diagnose_problem(
            &p,
            &Inputs::default(),
            "t".into(),
            &DiagnoseOptions::default(),
        )
        .unwrap();
        assert!(r.has_errors());
        assert!(
            r.findings
                .iter()
                .any(|f| f.code == "nonfinite_from_domain_violation")
        );
    }

    #[test]
    fn nan_the_interior_clamp_removes_is_a_warning() {
        // log(x), x in [0, 10], starting on the bound x = 0: log(0) = -inf at
        // the raw start, but the solver clamps the start to x = 0.01 first.
        let p = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Var(0),
            obj_constant: 0.0,
            constraints: vec![Expr::Unary(UnaryOp::Log, Box::new(Expr::Var(0)))],
            x_l: vec![0.0],
            x_u: vec![10.0],
            x0: vec![0.0],
            g_l: vec![-5.0],
            g_u: vec![1e19],
            var_names: vec![],
            con_names: vec![],
        })
        .unwrap();
        let r = diagnose_problem(
            &p,
            &Inputs::default(),
            "t".into(),
            &DiagnoseOptions::default(),
        )
        .unwrap();
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "nonfinite_from_pole")
            .expect("localized");
        assert_eq!(f.severity, Severity::Warning, "{}", f.why_it_matters);
        // Off the bound's reach of the clamp (x0 = 0 with no lower bound),
        // the same NaN is an error.
        let mut q = p.clone();
        q.x_l[0] = -1e19;
        let r = diagnose_problem(
            &q,
            &Inputs::default(),
            "t".into(),
            &DiagnoseOptions::default(),
        )
        .unwrap();
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "nonfinite_from_pole")
            .expect("localized");
        assert_eq!(f.severity, Severity::Error);
    }

    #[test]
    fn a_solution_point_is_not_clamped() {
        // The same log(x) at x = 0, but as a .sol point: the clamp models a
        // solve's start, not a point a solve reached, so no downgrade.
        let p = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Var(0),
            obj_constant: 0.0,
            constraints: vec![Expr::Unary(UnaryOp::Log, Box::new(Expr::Var(0)))],
            x_l: vec![0.0],
            x_u: vec![10.0],
            x0: vec![0.0],
            g_l: vec![-5.0],
            g_u: vec![1e19],
            var_names: vec![],
            con_names: vec![],
        })
        .unwrap();
        let inputs = Inputs {
            point: Some(("a solution".into(), vec![0.0])),
            multipliers: Some(vec![0.0]),
            is_solution: true,
            ..Inputs::default()
        };
        let r = diagnose_problem(&p, &inputs, "t".into(), &DiagnoseOptions::default()).unwrap();
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "nonfinite_from_pole")
            .expect("localized");
        assert_eq!(f.severity, Severity::Error);
        assert!(
            !r.findings
                .iter()
                .any(|f| f.code == "start_moved_by_bound_push")
        );
    }

    #[test]
    fn a_nonfinite_hessian_is_downgraded_only_if_finite_after_the_clamp() {
        // The Hessian check uses zero multipliers, so the pole goes in the
        // objective: min x0 + log(x1 - shift), x in [0, 10]^2.
        let mk = |shift: Number, x0: Vec<Number>| {
            NlProblem::from_expressions(NlProblemParts {
                minimize: true,
                objective: Expr::Binary(
                    BinOp::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Unary(
                        UnaryOp::Log,
                        Box::new(Expr::Binary(
                            BinOp::Sub,
                            Box::new(Expr::Var(1)),
                            Box::new(Expr::Const(shift)),
                        )),
                    )),
                ),
                obj_constant: 0.0,
                constraints: vec![],
                x_l: vec![0.0, 0.0],
                x_u: vec![10.0, 10.0],
                x0,
                g_l: vec![],
                g_u: vec![],
                var_names: vec![],
                con_names: vec![],
            })
            .unwrap()
        };
        let hess = |p: &NlProblem| {
            let r = diagnose_problem(
                p,
                &Inputs::default(),
                "t".into(),
                &DiagnoseOptions::default(),
            )
            .unwrap();
            r.findings
                .iter()
                .find(|f| f.code == "nonfinite_hessian_at_point")
                .map(|f| f.severity)
        };
        // x = (0, 5), shift 5: the clamp moves only x0, the pole at x1 = 5
        // stays, so a default solve hits it: an error.
        assert_eq!(hess(&mk(5.0, vec![0.0, 5.0])), Some(Severity::Error));
        // x = (0, 0), shift 0: the clamp moves x1 off the pole, and the
        // Hessian is finite there: a warning.
        assert_eq!(hess(&mk(0.0, vec![0.0, 0.0])), Some(Severity::Warning));
    }

    #[test]
    fn argv_parsing() {
        let a = parse_argv(&[
            "m.nl".into(),
            "--json".into(),
            "--rank-tol".into(),
            "1e-10".into(),
        ])
        .unwrap()
        .unwrap();
        assert!(a.json);
        assert_eq!(a.opts.rank_rel_tol, Some(1e-10));
        assert!(parse_argv(&[]).is_err());
        let b = parse_argv(&["m.nl".into(), "--convexity-tol".into(), "1e-6".into()])
            .unwrap()
            .unwrap();
        assert_eq!(b.opts.convexity_tol, 1e-6);
        assert!(parse_argv(&["m.nl".into(), "--bogus".into()]).is_err());
        assert!(
            parse_argv(&[
                "m.nl".into(),
                "--sol".into(),
                "a".into(),
                "--x0-file".into(),
                "b".into()
            ])
            .is_err()
        );
        assert!(parse_argv(&["--help".into()]).unwrap().is_none());
    }
}
