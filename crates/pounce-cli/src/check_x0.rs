//! `pounce check-x0 <problem.nl>` — starting-point preflight.
//!
//! # Why this exists
//!
//! A local NLP solver's fate is largely decided at iteration 0, but the
//! solver only reports starting-point trouble *after* it has tripped over
//! it (`Invalid_Number_Detected` mid-solve, immediate restoration, a slow
//! crawl caused by scaling). This subcommand evaluates the model once at
//! its starting point, before any solve, and reports what the initializer
//! and the first iteration will actually see:
//!
//! * **Non-finite evaluations** — NaN/inf in `f`, `∇f`, `g`, the Jacobian,
//!   or the Hessian at x0. These are fatal: the solve would abort.
//! * **Bound violations of x0** and components sitting exactly on a bound
//!   (the interior clamp will move both; see below).
//! * **Interior-clamp displacement** — the `bound_push` / `bound_frac`
//!   clamp (`DefaultIterateInitializer`) applied to x0, so "the solver
//!   silently moved my point" is visible up front.
//! * **Initial constraint violation** per row (infeasibility is fine for
//!   the IPM, but very large violations usually mean a wrong or missing
//!   starting point).
//! * **Derivative scale spread** — max/min nonzero magnitudes of `∇f` and
//!   the Jacobian at x0, the early-warning signal for scaling trouble.
//! * **Automatic scaling** — the objective and per-row factors
//!   `nlp_scaling_method=gradient-based` (the default) will pick *at this
//!   x0*, computed by the solver's own arithmetic; plus, for a `.nl` model,
//!   the coefficient magnitudes of its quadratic rows, which that sample
//!   cannot report. See [`ScalingPreview`].
//!
//! The checks are read-only and cost one evaluation of each callback:
//! `O(nnz)` work, no factorization, no solve.
//!
//! Verdict / exit code: `0` when the model evaluates cleanly at x0
//! (warnings allowed); `21` when an evaluation produced NaN/inf (the
//! solver would fail); `2` on a usage or I/O error.
//!
//! User-facing background: `docs/src/initialization.md`.

use crate::nl_reader;
use crate::verify::sha256;
use pounce_common::types::Number;
use pounce_nl::nl_scaling::{QuadRowCoef, quad_row_coefs};
use pounce_nlp::diagnostics::RowReport;
use pounce_nlp::diagnostics::preflight::{
    NLP_SCALING_MAX_GRADIENT, NLP_SCALING_MIN_VALUE, NonFinite, PreflightOptions, PreflightOutcome,
    RowScaleBlock, ScalingPreview, X0Override, check_tnlp_with_quadratics,
};
use pounce_nlp::tnlp::TNLP;
use std::path::PathBuf;
use std::process::ExitCode;

/// Parsed `check-x0` subcommand arguments.
#[derive(Debug, Clone)]
pub struct CheckX0Args {
    /// `.nl` path, or `None` when `--builtin` is used.
    pub nl: Option<PathBuf>,
    /// Built-in problem name (`--builtin rosenbrock`).
    pub builtin: Option<String>,
    /// Optional whitespace-separated file of `n` values overriding the
    /// model's starting point (`--x0-file`).
    pub x0_file: Option<PathBuf>,
    /// Violations above this are counted in `n_violated` (default 1e-6).
    pub feas_tol: Number,
    /// `bound_push` used for the clamp preview (default 1e-2).
    pub bound_push: Number,
    /// `bound_frac` used for the clamp preview (default 1e-2).
    pub bound_frac: Number,
    /// Max offenders listed per category (default 5).
    pub max_list: usize,
    /// `nlp_scaling_max_gradient` for the scaling preview (default 100).
    pub scaling_max_gradient: Number,
    /// Print the JSON report to stdout instead of the text report.
    pub json: bool,
    /// Also write the JSON report to this path.
    pub json_output: Option<PathBuf>,
}

impl Default for CheckX0Args {
    fn default() -> Self {
        CheckX0Args {
            nl: None,
            builtin: None,
            x0_file: None,
            feas_tol: 1e-6,
            bound_push: 1e-2,
            bound_frac: 1e-2,
            max_list: 5,
            scaling_max_gradient: NLP_SCALING_MAX_GRADIENT,
            json: false,
            json_output: None,
        }
    }
}

impl CheckX0Args {
    /// The numerical half of these arguments, as
    /// [`pounce_nlp::diagnostics::preflight`] wants them.
    ///
    /// `x0_file` is read here rather than in the core: the check takes a
    /// starting point, not a path, so a library caller supplies values
    /// directly and only the CLI has a file to open.
    fn preflight_options(&self) -> Result<PreflightOptions, String> {
        let x0 = match &self.x0_file {
            None => None,
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                let vals: Result<Vec<Number>, _> = text
                    .split_whitespace()
                    .map(|t| t.parse::<Number>())
                    .collect();
                let vals = vals.map_err(|e| format!("{}: bad value: {e}", path.display()))?;
                Some(X0Override {
                    x: vals,
                    source: format!("--x0-file {}", path.display()),
                })
            }
        };
        Ok(PreflightOptions {
            feas_tol: self.feas_tol,
            bound_push: self.bound_push,
            bound_frac: self.bound_frac,
            max_list: self.max_list,
            scaling_max_gradient: self.scaling_max_gradient,
            x0,
        })
    }
}

const USAGE: &str = "\
Usage: pounce check-x0 <problem.nl> [OPTIONS]
       pounce check-x0 --builtin <name> [OPTIONS]

Evaluate the model once at its starting point, before any solve, and
report what iteration 0 will see: NaN/inf evaluations (fatal), bound
violations of x0, how far the bound_push interior clamp will move the
point, initial constraint violation, derivative scale spread, and the
factors automatic (gradient-based) scaling will pick here.

Arguments:
  <problem.nl>           AMPL .nl problem (x0 from its initial-guess
                         segment; zeros for variables without one)

Options:
  --builtin <name>       check a built-in problem instead of a .nl file
  --x0-file <path>       override x0 with n whitespace-separated values
  --feas-tol <t>         constraint-violation report threshold (default 1e-6)
  --bound-push <v>       bound_push used for the clamp preview (default 1e-2)
  --bound-frac <v>       bound_frac used for the clamp preview (default 1e-2)
  --max-list <k>         max offenders listed per category (default 5)
  --scaling-max-gradient <v>
                         nlp_scaling_max_gradient for the scaling
                         preview (default 100)
  --json                 print the JSON report to stdout
  --json-output <path>   write the JSON report to <path>
  -h, --help             print this message

Exit code: 0 = model evaluates cleanly at x0 (warnings allowed),
21 = NaN/inf at x0 (a solve would abort), 2 = usage/IO error.";

/// Entry point dispatched from `main` when argv[1] == "check-x0".
pub fn run_from_argv(rest: &[String]) -> ExitCode {
    let args = match parse_argv(rest) {
        Ok(Some(a)) => a,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("pounce check-x0: {msg}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    run(&args)
}

fn parse_argv(rest: &[String]) -> Result<Option<CheckX0Args>, String> {
    let mut a = CheckX0Args::default();
    let mut positionals: Vec<PathBuf> = Vec::new();
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--builtin" => {
                let v = it.next().ok_or("--builtin requires a value")?;
                a.builtin = Some(v.clone());
            }
            "--x0-file" => {
                let v = it.next().ok_or("--x0-file requires a value")?;
                a.x0_file = Some(PathBuf::from(v));
            }
            "--feas-tol" => {
                let v = it.next().ok_or("--feas-tol requires a value")?;
                a.feas_tol = v.parse().map_err(|e| format!("--feas-tol: {e}"))?;
            }
            "--bound-push" => {
                let v = it.next().ok_or("--bound-push requires a value")?;
                a.bound_push = v.parse().map_err(|e| format!("--bound-push: {e}"))?;
            }
            "--bound-frac" => {
                let v = it.next().ok_or("--bound-frac requires a value")?;
                a.bound_frac = v.parse().map_err(|e| format!("--bound-frac: {e}"))?;
            }
            "--max-list" => {
                let v = it.next().ok_or("--max-list requires a value")?;
                a.max_list = v.parse().map_err(|e| format!("--max-list: {e}"))?;
            }
            "--scaling-max-gradient" => {
                let v = it.next().ok_or("--scaling-max-gradient requires a value")?;
                a.scaling_max_gradient = v
                    .parse()
                    .map_err(|e| format!("--scaling-max-gradient: {e}"))?;
                if a.scaling_max_gradient.is_nan() || a.scaling_max_gradient <= 0.0 {
                    return Err("--scaling-max-gradient must be positive".to_string());
                }
            }
            "--json" => a.json = true,
            "--json-output" => {
                let v = it.next().ok_or("--json-output requires a value")?;
                a.json_output = Some(PathBuf::from(v));
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag `{other}`"));
            }
            _ => positionals.push(PathBuf::from(arg)),
        }
    }
    match (positionals.len(), &a.builtin) {
        (0, Some(_)) => Ok(Some(a)),
        (1, None) => {
            a.nl = Some(positionals[0].clone());
            Ok(Some(a))
        }
        (0, None) => Err("expected a <problem.nl> argument or --builtin <name>".to_string()),
        _ => Err("expected exactly one of <problem.nl> or --builtin <name>".to_string()),
    }
}

pub fn run(args: &CheckX0Args) -> ExitCode {
    let outcome = match evaluate(args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("pounce check-x0: {msg}");
            return ExitCode::from(2);
        }
    };

    if args.json {
        println!("{}", report_json(&outcome));
    } else {
        print_report(&outcome);
    }
    if let Some(path) = &args.json_output {
        if let Err(e) = std::fs::write(path, report_json(&outcome).as_bytes()) {
            eprintln!(
                "pounce check-x0: failed to write report {}: {e}",
                path.display()
            );
            return ExitCode::from(2);
        }
        if !args.json {
            println!("  report: {}", path.display());
        }
    }

    if outcome.fatal {
        ExitCode::from(21)
    } else {
        ExitCode::SUCCESS
    }
}

/// A model loaded for preflight: the evaluator plus its provenance.
struct LoadedModel {
    tnlp: std::rc::Rc<std::cell::RefCell<dyn TNLP>>,
    var_names: Vec<String>,
    con_names: Vec<String>,
    nl_sha256: Option<String>,
    source: String,
    /// Quadratic-row coefficients, read off the `.nl` before the problem
    /// is consumed by the evaluator. Empty for a builtin, which has no
    /// file to read them from.
    quad_coefs: Vec<QuadRowCoef>,
}

fn load_model(args: &CheckX0Args) -> Result<LoadedModel, String> {
    if let Some(name) = &args.builtin {
        let tnlp = crate::builtin::lookup(name)
            .ok_or_else(|| format!("unknown builtin `{name}` (see `pounce --list-problems`)"))?;
        return Ok(LoadedModel {
            tnlp,
            var_names: Vec::new(),
            con_names: Vec::new(),
            nl_sha256: None,
            source: format!("builtin:{name}"),
            quad_coefs: Vec::new(),
        });
    }
    let path = args
        .nl
        .as_ref()
        .ok_or("expected a <problem.nl> argument or --builtin <name>")?;
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let sha = sha256::hex(&bytes);
    let prob = nl_reader::read_nl_file(path)?;
    let var_names = prob.var_names.clone();
    let con_names = prob.con_names.clone();
    // Read the quadratic coefficients before `prob` is consumed: they are
    // a property of the file, and the evaluator does not hand them back.
    let quad_coefs = quad_row_coefs(&prob);
    let t = nl_reader::NlTnlp::try_new(prob)?;
    Ok(LoadedModel {
        tnlp: std::rc::Rc::new(std::cell::RefCell::new(t)),
        var_names,
        con_names,
        nl_sha256: Some(sha),
        source: path.display().to_string(),
        quad_coefs,
    })
}

fn evaluate(args: &CheckX0Args) -> Result<PreflightOutcome, String> {
    let opts = args.preflight_options()?;
    let model = load_model(args)?;
    let mut tnlp = model.tnlp.borrow_mut();
    check_tnlp_with_quadratics(
        &mut *tnlp,
        &model.var_names,
        &model.con_names,
        model.nl_sha256.clone(),
        model.source.clone(),
        &model.quad_coefs[..],
        &opts,
    )
}

// ---------------------------------------------------------------------------
// Console + JSON rendering.
// ---------------------------------------------------------------------------

fn print_report(o: &PreflightOutcome) {
    println!("pounce check-x0 — starting-point preflight");
    println!(
        "  problem : {}  ({} vars, {} cons)",
        o.source, o.n_vars, o.n_cons
    );
    if let Some(sha) = &o.nl_sha256 {
        println!("            sha256:{sha}");
    }
    println!(
        "  x0      : {}{}",
        o.x0_source,
        if o.x0_all_zero { "  (all zeros)" } else { "" }
    );
    println!();

    println!("  evaluation at x0:");
    match o.objective {
        Some(v) if v.is_finite() => println!("    objective: {v:.10e}"),
        Some(v) => println!("    objective: {v}  <- NON-FINITE"),
        None => println!("    objective: EVALUATION FAILED"),
    }
    print_nonfinite("gradient", o.grad_nonfinite_count, &o.grad_nonfinite);
    print_nonfinite("constraints", o.g_nonfinite_count, &o.g_nonfinite);
    if o.jac_nonfinite_count > 0 {
        println!(
            "    Jacobian : {} non-finite entr{}",
            o.jac_nonfinite_count,
            if o.jac_nonfinite_count == 1 {
                "y"
            } else {
                "ies"
            }
        );
        for e in &o.jac_nonfinite {
            println!("        d{}/d{} = {}", e.row_name, e.col_name, e.value);
        }
    } else {
        println!("    Jacobian : finite");
    }
    match o.hess_nonfinite_count {
        Some(0) => println!("    Hessian  : finite (lambda=0)"),
        Some(k) => println!("    Hessian  : {k} non-finite entries (lambda=0)"),
        None => println!("    Hessian  : not checked (quasi-Newton or declined)"),
    }
    println!();

    println!("  x0 vs bounds:");
    println!(
        "    violations: {}  on-bound components: {}",
        o.n_bound_violations, o.n_on_bounds
    );
    for r in &o.bound_violations {
        println!(
            "        {}: value {:.6e} outside [{:.6e}, {:.6e}] by {:.3e}",
            r.name, r.value, r.lo, r.hi, r.violation
        );
    }
    println!(
        "    interior clamp moves {} component(s), max move {:.3e}",
        o.n_clamp_moved, o.max_clamp_move
    );
    for c in &o.clamp_moves {
        println!(
            "        {}: {:.6e} -> {:.6e}  (moved {:.3e})",
            c.name, c.from, c.to, c.distance
        );
    }
    println!();

    println!("  initial constraint violation:");
    println!(
        "    rows violated: {}  max violation: {:.3e}",
        o.n_con_violations, o.max_con_violation
    );
    for r in &o.con_violations {
        println!(
            "        {}: g = {:.6e}, bounds [{:.6e}, {:.6e}], violation {:.3e}",
            r.name, r.value, r.lo, r.hi, r.violation
        );
    }
    println!();

    println!("  derivative scale at x0:");
    println!(
        "    gradient: max |.| {:.3e}, min nonzero |.| {:.3e}",
        o.grad_spread.max_abs, o.grad_spread.min_abs_nonzero
    );
    println!(
        "    Jacobian: max |.| {:.3e}, min nonzero |.| {:.3e}",
        o.jac_spread.max_abs, o.jac_spread.min_abs_nonzero
    );
    println!();

    print_scaling(&o.scaling);

    if !o.warnings.is_empty() {
        println!("  warnings:");
        for w in &o.warnings {
            println!("    - {w}");
        }
        println!();
    }
    println!("  VERDICT: {}", o.verdict);
}

/// The `nlp_scaling_method=gradient-based` section of the text report.
fn print_scaling(s: &ScalingPreview) {
    println!(
        "  automatic scaling at x0 (nlp_scaling_method=gradient-based, \
         nlp_scaling_max_gradient={}):",
        s.max_gradient
    );
    println!(
        "    objective: ||grad f|| {:.3e} -> factor {:.3e}{}",
        s.max_grad_f,
        s.obj_scale,
        if s.obj_scale >= 1.0 {
            "  (below the cutoff: unscaled)"
        } else {
            ""
        }
    );
    for (label, b) in [("equalities", &s.c), ("inequalities", &s.d)] {
        if b.n_rows == 0 {
            continue;
        }
        if !b.fires {
            println!(
                "    {label:<12}: {} row(s), no row above the cutoff -> the whole \
                 block is unscaled",
                b.n_rows
            );
        } else {
            println!(
                "    {label:<12}: {} row(s), {} scaled down, min factor {:.3e}{}",
                b.n_rows,
                b.n_scaled,
                b.min_scale,
                if b.n_at_floor > 0 {
                    format!(
                        " ({} at the {:.0e} floor)",
                        b.n_at_floor, NLP_SCALING_MIN_VALUE
                    )
                } else {
                    String::new()
                }
            );
        }
        if b.n_zero_jac > 0 {
            println!(
                "                  {} row(s) have an all-zero Jacobian at x0 \
                 (the sample cannot scale them)",
                b.n_zero_jac
            );
        }
    }
    if s.n_quad_rows > 0 {
        println!(
            "    quadratic rows: {} recognized; {} left at factor 1.0, {} with a \
             zero Jacobian at x0",
            s.n_quad_rows, s.n_quad_unscaled, s.n_quad_zero_jac
        );
        println!(
            "                    worst |b|/||Q||_inf mismatch {:.3e}",
            s.max_quad_mismatch
        );
        for q in &s.quad_rows {
            println!(
                "        {}: ||Q||_inf {:.3e}, ||a||_inf {:.3e}, |b| {:.3e}, \
                 ||grad g(x0)||_inf {:.3e} -> factor {:.3e}, mismatch {:.3e}",
                q.name, q.curvature, q.linear, q.rhs, q.jac_at_x0, q.scale, q.mismatch
            );
        }
    }
    println!();
}

fn print_nonfinite(label: &str, count: usize, list: &[NonFinite]) {
    if count > 0 {
        println!(
            "    {label:<9}: {count} non-finite entr{}",
            if count == 1 { "y" } else { "ies" }
        );
        for e in list {
            println!("        {} = {}", e.name, e.value);
        }
    } else {
        println!("    {label:<9}: finite");
    }
}

fn block_json(b: &RowScaleBlock) -> serde_json::Value {
    serde_json::json!({
        "n_rows": b.n_rows,
        "fires": b.fires,
        "n_scaled": b.n_scaled,
        "min_factor": b.min_scale,
        "n_at_floor": b.n_at_floor,
        "n_zero_jacobian_at_x0": b.n_zero_jac,
    })
}

fn report_json(o: &PreflightOutcome) -> String {
    use serde_json::json;
    let row = |r: &RowReport| {
        json!({
            "index": r.index, "name": r.name, "value": r.value,
            "lower": r.lo, "upper": r.hi, "violation": r.violation,
        })
    };
    let nf =
        |e: &NonFinite| json!({"index": e.index, "name": e.name, "value": e.value.to_string()});
    let report = json!({
        "pounce_check_x0_version": 1,
        "schema": "pounce.check-x0/v1",
        "solver": format!("pounce {}", env!("CARGO_PKG_VERSION")),
        "problem": {
            "source": o.source,
            "sha256": o.nl_sha256,
            "n_vars": o.n_vars,
            "n_cons": o.n_cons,
        },
        "x0": { "source": o.x0_source, "all_zero": o.x0_all_zero },
        "evaluation": {
            "objective": o.objective.filter(|v| v.is_finite()),
            "objective_finite": o.objective.map(|v| v.is_finite()).unwrap_or(false),
            "grad_nonfinite_count": o.grad_nonfinite_count,
            "grad_nonfinite": o.grad_nonfinite.iter().map(nf).collect::<Vec<_>>(),
            "constraints_nonfinite_count": o.g_nonfinite_count,
            "constraints_nonfinite": o.g_nonfinite.iter().map(nf).collect::<Vec<_>>(),
            "jacobian_nonfinite_count": o.jac_nonfinite_count,
            "jacobian_nonfinite": o.jac_nonfinite.iter().map(|e| json!({
                "row": e.row, "col": e.col,
                "row_name": e.row_name, "col_name": e.col_name,
                "value": e.value.to_string(),
            })).collect::<Vec<_>>(),
            "hessian_nonfinite_count": o.hess_nonfinite_count,
        },
        "bounds": {
            "n_violations": o.n_bound_violations,
            "max_violation": o.max_bound_violation,
            "n_on_bounds": o.n_on_bounds,
            "worst": o.bound_violations.iter().map(row).collect::<Vec<_>>(),
        },
        "interior_clamp": {
            "n_moved": o.n_clamp_moved,
            "max_move": o.max_clamp_move,
            "worst": o.clamp_moves.iter().map(|c| json!({
                "index": c.index, "name": c.name,
                "from": c.from, "to": c.to, "distance": c.distance,
            })).collect::<Vec<_>>(),
        },
        "constraint_violation": {
            "n_violated": o.n_con_violations,
            "max_violation": o.max_con_violation,
            "worst": o.con_violations.iter().map(row).collect::<Vec<_>>(),
        },
        "derivative_scale": {
            "gradient": {
                "max_abs": o.grad_spread.max_abs,
                "min_abs_nonzero": o.grad_spread.min_abs_nonzero,
                "ratio": o.grad_spread.ratio,
            },
            "jacobian": {
                "max_abs": o.jac_spread.max_abs,
                "min_abs_nonzero": o.jac_spread.min_abs_nonzero,
                "ratio": o.jac_spread.ratio,
            },
        },
        "scaling": {
            "method": "gradient-based",
            "nlp_scaling_max_gradient": o.scaling.max_gradient,
            "nlp_scaling_min_value": NLP_SCALING_MIN_VALUE,
            "objective": {
                "max_abs_grad_f": o.scaling.max_grad_f,
                "factor": o.scaling.obj_scale,
            },
            "equalities": block_json(&o.scaling.c),
            "inequalities": block_json(&o.scaling.d),
            "quadratic_rows": {
                "n_rows": o.scaling.n_quad_rows,
                "n_unscaled": o.scaling.n_quad_unscaled,
                "n_zero_jacobian_at_x0": o.scaling.n_quad_zero_jac,
                "max_mismatch": o.scaling.max_quad_mismatch,
                "worst": o.scaling.quad_rows.iter().map(|q| json!({
                    "index": q.index, "name": q.name,
                    "curvature_inf_norm": q.curvature,
                    "linear_inf_norm": q.linear,
                    "rhs_abs": q.rhs,
                    "jacobian_inf_norm_at_x0": q.jac_at_x0,
                    "factor": q.scale,
                    "mismatch": q.mismatch,
                })).collect::<Vec<_>>(),
            },
        },
        "warnings": o.warnings,
        "fatal": o.fatal,
        "verdict": o.verdict,
    });
    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
}
#[cfg(test)]
mod tests {
    use super::*;

    /// `½·(4x₀² + 2x₁²) ≤ 1e5` written about the origin, plus a linear
    /// row so the model is not degenerate. Every variable is free and no
    /// initial guess is supplied, so `x0 = 0` and the quadratic row's
    /// Jacobian there is identically zero.
    const QUAD_AT_ORIGIN_NL: &str = "\
g3 0 1 0
 2 2 1 0 0
 1 0
 0 0
 2 2 2
 0 0 0 1
 0 0 0 0 0
 4 2
 0 0
 0 0 0 0 0
b
3
3
r
1 100000
1 7
C0
o54
2
o2
n0.5
o2
o2
n4.0
v0
v0
o2
n0.5
o2
o2
n2.0
v1
v1
C1
n0
O0 0
n0
k1
2
J0 2
0 0
1 0
J1 2
0 1
1 1
";

    fn quad_at_origin_outcome(opts: &PreflightOptions) -> PreflightOutcome {
        let prob = crate::nl_reader::parse_nl_text(QUAD_AT_ORIGIN_NL).expect("parse");
        let coefs = quad_row_coefs(&prob);
        let mut t = crate::nl_reader::NlTnlp::try_new(prob).expect("build");
        check_tnlp_with_quadratics(&mut t, &[], &[], None, "test".to_string(), &coefs[..], opts)
            .expect("check")
    }

    #[test]
    fn quadratic_row_written_about_the_origin_is_invisible_to_the_scaler() {
        let o = quad_at_origin_outcome(&PreflightOptions::default());
        let s = &o.scaling;
        assert_eq!(s.n_quad_rows, 1, "the ≤ row is the only quadratic one");
        assert_eq!(s.n_quad_zero_jac, 1, "∇g(0) = 0 for ½xᵀQx about the origin");
        assert_eq!(s.n_quad_unscaled, 1, "so the row keeps factor 1.0");

        let q = &s.quad_rows[0];
        // Q = diag(4, 2) ⇒ ‖Q‖_∞ = 4; no linear part; b = 1e5.
        assert!((q.curvature - 4.0).abs() < 1e-12);
        assert_eq!(q.linear, 0.0);
        assert!((q.rhs - 1.0e5).abs() < 1e-9);
        assert_eq!(q.jac_at_x0, 0.0);
        assert_eq!(q.scale, 1.0);
        assert!((q.mismatch - 2.5e4).abs() < 1e-6);

        // The mismatch is 2.5e4, well past the 1e2 threshold, so the
        // preflight says so rather than leaving it in the numbers.
        assert!(
            o.warnings
                .iter()
                .any(|w| w.contains("identically-zero Jacobian")),
            "expected the zero-Jacobian scaling warning, got {:?}",
            o.warnings
        );
    }

    #[test]
    fn moving_the_cutoff_moves_the_preview_but_not_the_blind_spot() {
        // At a cutoff of 1 the *linear* row (coefficients 1) still does
        // not exceed it, but the quadratic row is unreachable at any
        // cutoff: 100/0 and 1/0 both clamp to 1.0. The blind spot is not
        // a tuning problem.
        let opts = PreflightOptions {
            scaling_max_gradient: 1e-6,
            ..Default::default()
        };
        let o = quad_at_origin_outcome(&opts);
        assert!(o.scaling.d.fires, "the linear row is above a 1e-6 cutoff");
        assert_eq!(o.scaling.n_quad_unscaled, 1);
        assert_eq!(o.scaling.quad_rows[0].scale, 1.0);
    }
}
