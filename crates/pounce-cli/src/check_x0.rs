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
use crate::verify::{box_violation, is_finite_bound, name_at, sha256, RowReport};
use pounce_common::types::Number;
use pounce_nlp::tnlp::{BoundsInfo, SparsityRequest, StartingPoint, TNLP};
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
            json: false,
            json_output: None,
        }
    }
}

const USAGE: &str = "\
Usage: pounce check-x0 <problem.nl> [OPTIONS]
       pounce check-x0 --builtin <name> [OPTIONS]

Evaluate the model once at its starting point, before any solve, and
report what iteration 0 will see: NaN/inf evaluations (fatal), bound
violations of x0, how far the bound_push interior clamp will move the
point, initial constraint violation, and derivative scale spread.

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

/// One non-finite evaluation entry.
#[derive(Debug, Clone)]
pub struct NonFinite {
    pub index: usize,
    pub name: String,
    pub value: Number,
}

/// One Jacobian/Hessian non-finite entry (row/col in matrix coordinates).
#[derive(Debug, Clone)]
pub struct NonFiniteEntry {
    pub row: usize,
    pub col: usize,
    pub row_name: String,
    pub col_name: String,
    pub value: Number,
}

/// One interior-clamp displacement entry.
#[derive(Debug, Clone)]
pub struct ClampMove {
    pub index: usize,
    pub name: String,
    pub from: Number,
    pub to: Number,
    pub distance: Number,
}

/// Max/min-nonzero magnitude summary of a derivative array at x0.
#[derive(Debug, Clone, Default)]
pub struct ScaleSpread {
    pub max_abs: Number,
    pub min_abs_nonzero: Number,
    /// `max_abs / min_abs_nonzero`, or 0 when there are no nonzeros.
    pub ratio: Number,
}

/// The fully-evaluated preflight result.
#[derive(Debug)]
pub struct CheckX0Outcome {
    pub n_vars: usize,
    pub n_cons: usize,
    pub nl_sha256: Option<String>,
    pub source: String,
    pub x0_source: String,
    pub x0_all_zero: bool,
    pub objective: Option<Number>,
    // non-finite scans (counts are totals; lists are capped at max_list)
    pub grad_nonfinite: Vec<NonFinite>,
    pub grad_nonfinite_count: usize,
    pub g_nonfinite: Vec<NonFinite>,
    pub g_nonfinite_count: usize,
    pub jac_nonfinite: Vec<NonFiniteEntry>,
    pub jac_nonfinite_count: usize,
    /// `None` when the TNLP declines exact Hessians (quasi-Newton).
    pub hess_nonfinite_count: Option<usize>,
    // x0 vs bounds
    pub bound_violations: Vec<RowReport>,
    pub n_bound_violations: usize,
    pub max_bound_violation: Number,
    pub n_on_bounds: usize,
    // interior-clamp preview
    pub clamp_moves: Vec<ClampMove>,
    pub n_clamp_moved: usize,
    pub max_clamp_move: Number,
    // initial constraint violation
    pub con_violations: Vec<RowReport>,
    pub n_con_violations: usize,
    pub max_con_violation: Number,
    // derivative scale spread
    pub grad_spread: ScaleSpread,
    pub jac_spread: ScaleSpread,
    // rollup
    pub warnings: Vec<String>,
    pub fatal: bool,
    pub verdict: &'static str,
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
    let t = nl_reader::NlTnlp::try_new(prob)?;
    Ok(LoadedModel {
        tnlp: std::rc::Rc::new(std::cell::RefCell::new(t)),
        var_names,
        con_names,
        nl_sha256: Some(sha),
        source: path.display().to_string(),
    })
}

fn evaluate(args: &CheckX0Args) -> Result<CheckX0Outcome, String> {
    let model = load_model(args)?;
    let mut tnlp = model.tnlp.borrow_mut();
    check_tnlp(
        &mut *tnlp,
        &model.var_names,
        &model.con_names,
        model.nl_sha256.clone(),
        model.source.clone(),
        args,
    )
}

/// The core preflight over any TNLP. Public so the debugger / tests can
/// reuse it without going through a file.
pub fn check_tnlp(
    tnlp: &mut dyn TNLP,
    var_names: &[String],
    con_names: &[String],
    nl_sha256: Option<String>,
    source: String,
    args: &CheckX0Args,
) -> Result<CheckX0Outcome, String> {
    let info = tnlp.get_nlp_info().ok_or("get_nlp_info failed")?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    let nnz = info.nnz_jac_g.max(0) as usize;
    let nnz_h = info.nnz_h_lag.max(0) as usize;
    let fortran = matches!(info.index_style, pounce_nlp::tnlp::IndexStyle::Fortran);
    let off = if fortran { 1usize } else { 0usize };

    // --- bounds ---
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return Err("get_bounds_info failed".to_string());
    }

    // --- starting point ---
    let mut x = vec![0.0; n];
    let (mut zl_buf, mut zu_buf, mut lam_buf) = (vec![0.0; n], vec![0.0; n], vec![0.0; m]);
    let x0_source = if let Some(path) = &args.x0_file {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let vals: Result<Vec<Number>, _> = text
            .split_whitespace()
            .map(|t| t.parse::<Number>())
            .collect();
        let vals = vals.map_err(|e| format!("{}: bad value: {e}", path.display()))?;
        if vals.len() != n {
            return Err(format!(
                "{} has {} values but the problem has {n} variables",
                path.display(),
                vals.len()
            ));
        }
        x.copy_from_slice(&vals);
        format!("--x0-file {}", path.display())
    } else {
        if !tnlp.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut zl_buf,
            z_u: &mut zu_buf,
            init_lambda: false,
            lambda: &mut lam_buf,
        }) {
            return Err("get_starting_point failed".to_string());
        }
        "model".to_string()
    };
    let x0_all_zero = n > 0 && x.iter().all(|v| *v == 0.0);

    // --- evaluations at x0 ---
    let objective = tnlp.eval_f(&x, true);
    let obj_finite = objective.map(|v| v.is_finite()).unwrap_or(false);

    let mut grad_f = vec![0.0; n];
    let grad_ok = tnlp.eval_grad_f(&x, false, &mut grad_f);
    let (grad_nonfinite, grad_nonfinite_count) =
        scan_nonfinite(&grad_f, var_names, 'x', args.max_list, grad_ok);

    let mut g = vec![0.0; m];
    let g_ok = m == 0 || tnlp.eval_g(&x, false, &mut g);
    let (g_nonfinite, g_nonfinite_count) = scan_nonfinite(&g, con_names, 'c', args.max_list, g_ok);

    // Jacobian: structure then values.
    let mut irow = vec![0i32; nnz];
    let mut jcol = vec![0i32; nnz];
    let mut jval = vec![0.0; nnz];
    let mut jac_ok = nnz == 0;
    if nnz > 0 {
        jac_ok = tnlp.eval_jac_g(
            Some(&x),
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) && tnlp.eval_jac_g(
            Some(&x),
            false,
            SparsityRequest::Values { values: &mut jval },
        );
    }
    let mut jac_nonfinite = Vec::new();
    let mut jac_nonfinite_count = 0usize;
    if jac_ok {
        for k in 0..nnz {
            if !jval[k].is_finite() {
                jac_nonfinite_count += 1;
                if jac_nonfinite.len() < args.max_list {
                    let row = (irow[k] as usize).wrapping_sub(off);
                    let col = (jcol[k] as usize).wrapping_sub(off);
                    jac_nonfinite.push(NonFiniteEntry {
                        row,
                        col,
                        row_name: name_at(con_names, row, 'c'),
                        col_name: name_at(var_names, col, 'x'),
                        value: jval[k],
                    });
                }
            }
        }
    } else if nnz > 0 {
        jac_nonfinite_count = usize::MAX; // "evaluation itself failed"
    }

    // Hessian of the Lagrangian at (x0, lambda=0, obj_factor=1) — catches
    // second-derivative domain errors. Optional: quasi-Newton TNLPs decline.
    let hess_nonfinite_count = if nnz_h > 0 {
        let mut hrow = vec![0i32; nnz_h];
        let mut hcol = vec![0i32; nnz_h];
        let mut hval = vec![0.0; nnz_h];
        let lambda0 = vec![0.0; m];
        let ok = tnlp.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut hrow,
                jcol: &mut hcol,
            },
        ) && tnlp.eval_h(
            Some(&x),
            false,
            1.0,
            Some(&lambda0),
            true,
            SparsityRequest::Values { values: &mut hval },
        );
        if ok {
            Some(hval.iter().filter(|v| !v.is_finite()).count())
        } else {
            None
        }
    } else {
        None
    };

    // --- x0 vs bounds ---
    let mut bound_violations: Vec<RowReport> = Vec::new();
    let mut n_bound_violations = 0usize;
    let mut max_bound_violation = 0.0_f64;
    let mut n_on_bounds = 0usize;
    for j in 0..n {
        let viol = box_violation(x[j], x_l[j], x_u[j]);
        if viol > args.feas_tol {
            n_bound_violations += 1;
            max_bound_violation = max_bound_violation.max(viol);
            push_worst(
                &mut bound_violations,
                RowReport {
                    index: j,
                    name: name_at(var_names, j, 'x'),
                    value: x[j],
                    lo: x_l[j],
                    hi: x_u[j],
                    violation: viol,
                },
                args.max_list,
            );
        }
        if x[j].is_finite() {
            let at_lo =
                is_finite_bound(x_l[j]) && (x[j] - x_l[j]).abs() <= 1e-8 * (1.0 + x_l[j].abs());
            let at_hi =
                is_finite_bound(x_u[j]) && (x_u[j] - x[j]).abs() <= 1e-8 * (1.0 + x_u[j].abs());
            if at_lo || at_hi {
                n_on_bounds += 1;
            }
        }
    }

    // --- interior-clamp preview (DefaultIterateInitializer::push_to_interior) ---
    let mut clamp_moves: Vec<ClampMove> = Vec::new();
    let mut n_clamp_moved = 0usize;
    let mut max_clamp_move = 0.0_f64;
    for j in 0..n {
        if !x[j].is_finite() {
            continue;
        }
        let to = clamp_to_interior(x[j], x_l[j], x_u[j], args.bound_push, args.bound_frac);
        let d = (to - x[j]).abs();
        if d > 0.0 {
            n_clamp_moved += 1;
            max_clamp_move = max_clamp_move.max(d);
            if clamp_moves.len() < args.max_list
                || clamp_moves.last().map(|w| d > w.distance).unwrap_or(false)
            {
                clamp_moves.push(ClampMove {
                    index: j,
                    name: name_at(var_names, j, 'x'),
                    from: x[j],
                    to,
                    distance: d,
                });
                clamp_moves.sort_by(|a, b| {
                    b.distance
                        .partial_cmp(&a.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                clamp_moves.truncate(args.max_list);
            }
        }
    }

    // --- initial constraint violation ---
    let mut con_violations: Vec<RowReport> = Vec::new();
    let mut n_con_violations = 0usize;
    let mut max_con_violation = 0.0_f64;
    if g_ok {
        for i in 0..m {
            let viol = box_violation(g[i], g_l[i], g_u[i]);
            if viol > args.feas_tol {
                n_con_violations += 1;
                if viol.is_finite() {
                    max_con_violation = max_con_violation.max(viol);
                }
                push_worst(
                    &mut con_violations,
                    RowReport {
                        index: i,
                        name: name_at(con_names, i, 'c'),
                        value: g[i],
                        lo: g_l[i],
                        hi: g_u[i],
                        violation: viol,
                    },
                    args.max_list,
                );
            }
        }
    }

    // --- derivative scale spread ---
    let grad_spread = scale_spread(grad_f.iter().copied());
    let jac_spread = scale_spread(jval.iter().copied());

    // --- warnings + verdict ---
    let mut warnings = Vec::new();
    let eval_failed = !grad_ok || !g_ok || (!jac_ok && nnz > 0) || objective.is_none();
    let nonfinite_total = grad_nonfinite_count.min(usize::MAX - 1)
        + g_nonfinite_count.min(usize::MAX - 1)
        + if jac_nonfinite_count == usize::MAX {
            0
        } else {
            jac_nonfinite_count
        }
        + hess_nonfinite_count.unwrap_or(0)
        + usize::from(!obj_finite && objective.is_some());
    let fatal = eval_failed || nonfinite_total > 0;
    if eval_failed {
        warnings.push(
            "an evaluation callback failed outright at the starting point; \
             the solver cannot start from this x0"
                .to_string(),
        );
    }
    if nonfinite_total > 0 {
        warnings.push(format!(
            "{nonfinite_total} non-finite value(s) at the starting point; a solve \
             would abort with Invalid_Number_Detected. The interior clamp only \
             repairs bound violations, not domain errors — move x0 into the \
             domain or add bounds that keep it there"
        ));
    }
    if x0_all_zero {
        warnings.push(
            "the starting point is all zeros: the model supplies no initial \
             guess (or an explicitly zero one)"
                .to_string(),
        );
    }
    if n_bound_violations > 0 {
        warnings.push(format!(
            "x0 violates {n_bound_violations} variable bound(s) (max {max_bound_violation:.3e}); \
             the initializer will clamp them inside"
        ));
    }
    if n_on_bounds > 0 {
        warnings.push(format!(
            "{n_on_bounds} component(s) of x0 sit exactly on a bound and will be \
             pushed into the interior (bound_push={:.1e}); if x0 is a previous \
             solution, use the warm-start recipe (warm_start_init_point=yes with \
             tightened warm_start_bound_push/_frac)",
            args.bound_push
        ));
    }
    if max_con_violation > 1e4 {
        warnings.push(format!(
            "very large initial infeasibility (max constraint violation \
             {max_con_violation:.3e}); consider a better starting point or \
             least_square_init_primal=yes"
        ));
    }
    for (label, s) in [("gradient", &grad_spread), ("Jacobian", &jac_spread)] {
        if s.ratio > 1e8 || s.max_abs > 1e8 {
            warnings.push(format!(
                "{label} magnitudes at x0 span a large range (max {:.3e}, min \
                 nonzero {:.3e}); see the scaling reference page",
                s.max_abs, s.min_abs_nonzero
            ));
        }
    }

    let verdict = if fatal {
        "FATAL"
    } else if warnings.is_empty() {
        "CLEAN"
    } else {
        "WARNINGS"
    };

    Ok(CheckX0Outcome {
        n_vars: n,
        n_cons: m,
        nl_sha256,
        source,
        x0_source,
        x0_all_zero,
        objective,
        grad_nonfinite,
        grad_nonfinite_count,
        g_nonfinite,
        g_nonfinite_count,
        jac_nonfinite,
        jac_nonfinite_count: if jac_nonfinite_count == usize::MAX {
            0
        } else {
            jac_nonfinite_count
        },
        hess_nonfinite_count,
        bound_violations,
        n_bound_violations,
        max_bound_violation,
        n_on_bounds,
        clamp_moves,
        n_clamp_moved,
        max_clamp_move,
        con_violations,
        n_con_violations,
        max_con_violation,
        grad_spread,
        jac_spread,
        warnings,
        fatal,
        verdict,
    })
}

/// The per-component interior clamp from
/// `DefaultIterateInitializer::push_to_interior` (see
/// `crates/pounce-algorithm/src/init/default.rs` and
/// `docs/src/initialization.md`).
pub fn clamp_to_interior(
    x: Number,
    lo: Number,
    hi: Number,
    bound_push: Number,
    bound_frac: Number,
) -> Number {
    match (is_finite_bound(lo), is_finite_bound(hi)) {
        (true, true) => {
            let span = hi - lo;
            let p_l = (bound_push * lo.abs().max(1.0)).min(bound_frac * span);
            let p_u = (bound_push * hi.abs().max(1.0)).min(bound_frac * span);
            x.max(lo + p_l).min(hi - p_u)
        }
        (true, false) => x.max(lo + bound_push * lo.abs().max(1.0)),
        (false, true) => x.min(hi - bound_push * hi.abs().max(1.0)),
        (false, false) => x,
    }
}

fn scan_nonfinite(
    values: &[Number],
    names: &[String],
    kind: char,
    cap: usize,
    eval_ok: bool,
) -> (Vec<NonFinite>, usize) {
    if !eval_ok {
        return (Vec::new(), 0);
    }
    let mut out = Vec::new();
    let mut count = 0usize;
    for (i, v) in values.iter().enumerate() {
        if !v.is_finite() {
            count += 1;
            if out.len() < cap {
                out.push(NonFinite {
                    index: i,
                    name: name_at(names, i, kind),
                    value: *v,
                });
            }
        }
    }
    (out, count)
}

/// Keep the `cap` worst entries by violation, descending.
fn push_worst(list: &mut Vec<RowReport>, r: RowReport, cap: usize) {
    list.push(r);
    list.sort_by(|a, b| {
        b.violation
            .partial_cmp(&a.violation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    list.truncate(cap);
}

fn scale_spread(values: impl Iterator<Item = Number>) -> ScaleSpread {
    let mut max_abs = 0.0_f64;
    let mut min_abs = Number::INFINITY;
    for v in values {
        let a = v.abs();
        if a.is_finite() && a > 0.0 {
            max_abs = max_abs.max(a);
            min_abs = min_abs.min(a);
        }
    }
    if max_abs == 0.0 {
        ScaleSpread::default()
    } else {
        ScaleSpread {
            max_abs,
            min_abs_nonzero: min_abs,
            ratio: max_abs / min_abs,
        }
    }
}

// ---------------------------------------------------------------------------
// Console + JSON rendering.
// ---------------------------------------------------------------------------

fn print_report(o: &CheckX0Outcome) {
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

    if !o.warnings.is_empty() {
        println!("  warnings:");
        for w in &o.warnings {
            println!("    - {w}");
        }
        println!();
    }
    println!("  VERDICT: {}", o.verdict);
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

fn report_json(o: &CheckX0Outcome) -> String {
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
        "warnings": o.warnings,
        "fatal": o.fatal,
        "verdict": o.verdict,
    });
    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
    use pounce_nlp::tnlp::{IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution};

    /// min 1/x0 + x1  s.t. x0 + x1 = 1, with x0 starting AT zero — the
    /// canonical Invalid_Number_Detected trap.
    struct DomainTrap {
        x0: Vec<Number>,
    }

    impl TNLP for DomainTrap {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 2,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[0.0, NLP_LOWER_BOUND_INF]);
            b.x_u
                .copy_from_slice(&[NLP_UPPER_BOUND_INF, NLP_UPPER_BOUND_INF]);
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            if sp.init_x {
                sp.x.copy_from_slice(&self.x0);
            }
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(1.0 / x[0] + x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f[0] = -1.0 / (x[0] * x[0]);
            grad_f[1] = 1.0;
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
    }

    fn check(x0: Vec<Number>) -> CheckX0Outcome {
        let mut t = DomainTrap { x0 };
        check_tnlp(
            &mut t,
            &[],
            &[],
            None,
            "test".into(),
            &CheckX0Args::default(),
        )
        .expect("check")
    }

    #[test]
    fn nan_at_x0_is_fatal() {
        // x0[0] = 0 → f = 1/0 = inf, grad[0] = -inf.
        let o = check(vec![0.0, 0.0]);
        assert!(o.fatal);
        assert_eq!(o.verdict, "FATAL");
        assert!(o.grad_nonfinite_count >= 1);
        assert!(o.x0_all_zero);
    }

    #[test]
    fn clean_interior_point_passes() {
        let o = check(vec![0.5, 0.5]);
        assert!(!o.fatal);
        assert_eq!(o.n_bound_violations, 0);
        // x0 + x1 = 1 exactly: feasible.
        assert_eq!(o.n_con_violations, 0);
        assert_eq!(o.verdict, "CLEAN");
        assert!((o.objective.unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn on_bound_component_is_flagged_and_clamped() {
        // x0[0] = 1e-12 is (numerically) on its lower bound 0; the clamp
        // moves it to ~bound_push = 1e-2 (span is infinite: one-sided).
        let o = check(vec![1e-12, 1.0]);
        assert!(o.n_on_bounds >= 1);
        assert!(o.n_clamp_moved >= 1);
        assert!((o.max_clamp_move - 1e-2).abs() < 1e-9);
        assert!(o
            .warnings
            .iter()
            .any(|w| w.contains("warm_start_bound_push")));
    }

    #[test]
    fn bound_violation_reported() {
        let o = check(vec![-3.0, 1.0]);
        assert_eq!(o.n_bound_violations, 1);
        assert!((o.max_bound_violation - 3.0).abs() < 1e-12);
        // clamp brings it inside: from -3 to lo + push
        assert!(o.n_clamp_moved >= 1);
    }

    #[test]
    fn infeasible_start_is_not_fatal() {
        let o = check(vec![5.0, 5.0]);
        assert!(!o.fatal);
        assert_eq!(o.n_con_violations, 1);
        assert!((o.max_con_violation - 9.0).abs() < 1e-12);
    }

    #[test]
    fn clamp_formula_matches_default_initializer() {
        // Two-sided [1, 5], bound_push=bound_frac=1e-2:
        // p_l = min(1e-2*1, 1e-2*4) = 0.01 → 1.0 clamps to 1.01.
        assert!((clamp_to_interior(1.0, 1.0, 5.0, 1e-2, 1e-2) - 1.01).abs() < 1e-15);
        // Interior stays put.
        assert_eq!(clamp_to_interior(3.0, 1.0, 5.0, 1e-2, 1e-2), 3.0);
        // Free variable untouched.
        assert_eq!(
            clamp_to_interior(-7.0, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, 1e-2, 1e-2),
            -7.0
        );
        // Upper one-sided: hi=100 → push = 1e-2*100 = 1 → 100 → 99.
        assert!(
            (clamp_to_interior(100.0, NLP_LOWER_BOUND_INF, 100.0, 1e-2, 1e-2) - 99.0).abs() < 1e-12
        );
    }

    #[test]
    fn scale_spread_ignores_zeros_and_nonfinite() {
        let s = scale_spread(vec![0.0, 1e-6, 1e3, Number::NAN].into_iter());
        assert!((s.max_abs - 1e3).abs() < 1e-9);
        assert!((s.min_abs_nonzero - 1e-6).abs() < 1e-18);
        assert!((s.ratio - 1e9).abs() / 1e9 < 1e-9);
    }
}
