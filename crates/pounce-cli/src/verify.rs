//! `pounce verify <problem.nl> <claim.sol>` — independent solution checker.
//!
//! # Why this exists
//!
//! When pounce is a *tool an agent calls*, the agent should never be the
//! thing you trust for "the solution satisfies the constraints." Trust
//! belongs to a small, deterministic checker that re-derives the answer
//! from the **canonical** problem — not from the agent's narration and not
//! even from the solver's own exit string. Optimization is the rare setting
//! where this is cheap: a claimed `x*` is just numbers, and feasibility is
//! one constraint evaluation (`g_l ≤ g(x*) ≤ g_u`, `x_l ≤ x* ≤ x_u`),
//! `O(nnz)` work with no resolve.
//!
//! `pounce verify` loads the canonical `.nl`, reads a claimed `.sol`, and
//! reports the worst constraint/bound violation (and, when the `.sol`
//! carries constraint duals, a first-order/KKT stationarity residual). It
//! defends the three agent-workflow failure modes:
//!
//! * **fabrication** ("here's a solution that looks like pounce ran") —
//!   invented numbers fail the residual check against the real model;
//! * **ignoring the solver** — a downstream consumer gates on the receipt's
//!   `verified: true` plus the problem hash, not on prose;
//! * **solving the wrong problem** (dropping/relaxing a constraint to dodge
//!   infeasibility) — the check runs against the *canonical* constraints
//!   and bounds, so a point that is only feasible for a relaxed model is
//!   caught here.
//!
//! The JSON receipt content-addresses both inputs by SHA-256 so a consumer
//! can confirm *which* problem was verified. When the `POUNCE_VERIFY_KEY`
//! environment variable holds a secret the agent does not have, the receipt
//! is additionally signed with HMAC-SHA256 over a float-free preimage (see
//! [`signing_preimage`]) — so an agent cannot mint a receipt that a consumer
//! holding the key will accept. The consumer recomputes the HMAC over the
//! same preimage and compares.
//!
//! Verdict / exit code: `0` when every violation is within tolerance
//! (`FEASIBLE`); `20` when a violation exceeds tolerance (`INFEASIBLE`);
//! `2` on a usage or I/O error. Optimality is reported but, by default,
//! does not gate — feasibility is the rigorous, sign-convention-independent
//! guarantee; pass `--require-optimal` to also gate on the stationarity
//! residual.

use crate::nl_reader;
use pounce_common::types::{Number, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_nlp::tnlp::{BoundsInfo, IndexStyle, SparsityRequest, TNLP};
use std::path::PathBuf;
use std::process::ExitCode;

/// Parsed `verify` subcommand arguments.
#[derive(Debug, Clone)]
pub struct VerifyArgs {
    pub nl: PathBuf,
    pub sol: PathBuf,
    /// Max `|violation|` of any constraint or bound still called feasible.
    pub feas_tol: Number,
    /// Max stationarity residual still called first-order optimal.
    pub opt_tol: Number,
    /// `--json-output PATH` — write the machine-readable receipt to PATH.
    pub json_output: Option<PathBuf>,
    /// `--require-optimal` — also gate the exit code on the stationarity
    /// residual (needs duals in the `.sol`).
    pub require_optimal: bool,
}

impl Default for VerifyArgs {
    fn default() -> Self {
        VerifyArgs {
            nl: PathBuf::new(),
            sol: PathBuf::new(),
            feas_tol: 1e-6,
            opt_tol: 1e-6,
            json_output: None,
            require_optimal: false,
        }
    }
}

const USAGE: &str = "\
Usage: pounce verify <problem.nl> <claim.sol> [OPTIONS]

Independently check that the solution in <claim.sol> satisfies the
constraints and bounds of the canonical problem <problem.nl>. Re-derives
feasibility from the model itself — it does not trust the .sol's status
line or rerun the solver.

Arguments:
  <problem.nl>            canonical AMPL .nl problem (the source of truth)
  <claim.sol>            claimed AMPL .sol solution to check

Options:
  --feas-tol <t>         feasibility tolerance (default 1e-6)
  --opt-tol <t>          stationarity tolerance (default 1e-6)
  --require-optimal      also fail if the KKT stationarity residual
                         exceeds --opt-tol (needs duals in the .sol)
  --json-output <path>   write a JSON verification receipt to <path>
  -h, --help             print this message

Exit code: 0 = verified feasible, 20 = violation exceeds tolerance,
2 = usage/IO error.";

/// Entry point dispatched from `main` when argv[1] == "verify".
pub fn run_from_argv(rest: &[String]) -> ExitCode {
    let args = match parse_verify_argv(rest) {
        Ok(Some(a)) => a,
        Ok(None) => {
            // help was requested
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("pounce verify: {msg}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    run(&args)
}

fn parse_verify_argv(rest: &[String]) -> Result<Option<VerifyArgs>, String> {
    let mut a = VerifyArgs::default();
    let mut positionals: Vec<PathBuf> = Vec::new();
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--feas-tol" => {
                let v = it.next().ok_or("--feas-tol requires a value")?;
                a.feas_tol = v.parse().map_err(|e| format!("--feas-tol: {e}"))?;
            }
            "--opt-tol" => {
                let v = it.next().ok_or("--opt-tol requires a value")?;
                a.opt_tol = v.parse().map_err(|e| format!("--opt-tol: {e}"))?;
            }
            "--require-optimal" => a.require_optimal = true,
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
    match positionals.len() {
        0 | 1 => Err("expected two positional arguments: <problem.nl> <claim.sol>".to_string()),
        2 => {
            a.nl = positionals[0].clone();
            a.sol = positionals[1].clone();
            Ok(Some(a))
        }
        n => Err(format!("expected 2 positional arguments, got {n}")),
    }
}

/// The fully-evaluated verification result. Serialized to the JSON
/// receipt and rendered to the console.
#[derive(Debug)]
pub struct VerifyOutcome {
    pub n_vars: usize,
    pub n_cons: usize,
    pub nl_sha256: String,
    pub sol_sha256: String,
    pub solve_result_num: Option<i32>,
    pub feas_tol: Number,
    pub opt_tol: Number,
    // feasibility
    pub max_con_violation: Number,
    pub worst_con: Option<RowReport>,
    pub max_bound_violation: Number,
    pub worst_bound: Option<RowReport>,
    pub feasible: bool,
    // optimality (only when duals supplied)
    pub objective: Option<Number>,
    pub duals_present: bool,
    pub stationarity: Option<Number>,
    pub dual_sign: Option<i32>,
    pub complementarity: Option<Number>,
    pub optimal: Option<bool>,
    // final
    pub verified: bool,
}

#[derive(Debug, Clone)]
pub struct RowReport {
    pub index: usize,
    pub name: String,
    pub value: Number,
    pub lo: Number,
    pub hi: Number,
    pub violation: Number,
}

fn is_finite_bound(b: Number) -> bool {
    b > NLP_LOWER_BOUND_INF && b < NLP_UPPER_BOUND_INF
}

/// `g_l ≤ v ≤ g_u` violation: how far `v` is outside the box, 0 if inside.
fn box_violation(v: Number, lo: Number, hi: Number) -> Number {
    let below = if is_finite_bound(lo) {
        lo - v
    } else {
        Number::NEG_INFINITY
    };
    let above = if is_finite_bound(hi) {
        v - hi
    } else {
        Number::NEG_INFINITY
    };
    below.max(above).max(0.0)
}

pub fn run(args: &VerifyArgs) -> ExitCode {
    let outcome = match evaluate(args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("pounce verify: {msg}");
            return ExitCode::from(2);
        }
    };
    print_report(args, &outcome);

    if let Some(path) = &args.json_output {
        let json = receipt_json(args, &outcome);
        if let Err(e) = std::fs::write(path, json.as_bytes()) {
            eprintln!(
                "pounce verify: failed to write receipt {}: {e}",
                path.display()
            );
            return ExitCode::from(2);
        }
        let signed = std::env::var(KEY_ENV)
            .map(|k| !k.is_empty())
            .unwrap_or(false);
        println!(
            "  receipt: {}{}",
            path.display(),
            if signed {
                "  (signed: HMAC-SHA256)"
            } else {
                ""
            }
        );
    }

    if outcome.verified {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(20)
    }
}

fn evaluate(args: &VerifyArgs) -> Result<VerifyOutcome, String> {
    // --- read + hash the two inputs (content-address the receipt) ---
    let nl_bytes =
        std::fs::read(&args.nl).map_err(|e| format!("cannot read {}: {e}", args.nl.display()))?;
    let sol_bytes =
        std::fs::read(&args.sol).map_err(|e| format!("cannot read {}: {e}", args.sol.display()))?;
    let nl_sha256 = sha256::hex(&nl_bytes);
    let sol_sha256 = sha256::hex(&sol_bytes);

    // --- canonical problem ---
    let prob = nl_reader::read_nl_file(&args.nl)?;
    let n = prob.n;
    let m = prob.m;
    let con_names = prob.con_names.clone();
    let var_names = prob.var_names.clone();
    let mut tnlp = nl_reader::NlTnlp::new(prob);

    let info = tnlp
        .get_nlp_info()
        .ok_or("get_nlp_info failed on the .nl")?;
    let nnz = info.nnz_jac_g.max(0) as usize;
    let fortran = matches!(info.index_style, IndexStyle::Fortran);

    // --- claimed solution ---
    let sol_text = String::from_utf8_lossy(&sol_bytes);
    let parsed = parse_sol(&sol_text)?;
    if parsed.x.len() != n {
        return Err(format!(
            "solution has {} primal values but the problem has {n} variables \
             (is this the right .sol for this .nl?)",
            parsed.x.len()
        ));
    }
    let x = parsed.x;
    let duals_present = !parsed.lambda.is_empty();
    if duals_present && parsed.lambda.len() != m {
        return Err(format!(
            "solution carries {} dual values but the problem has {m} constraints",
            parsed.lambda.len()
        ));
    }

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

    // --- bound feasibility ---
    let mut max_bound_violation = 0.0_f64;
    let mut worst_bound: Option<RowReport> = None;
    for j in 0..n {
        let viol = box_violation(x[j], x_l[j], x_u[j]);
        if viol > max_bound_violation {
            max_bound_violation = viol;
            worst_bound = Some(RowReport {
                index: j,
                name: name_at(&var_names, j, 'x'),
                value: x[j],
                lo: x_l[j],
                hi: x_u[j],
                violation: viol,
            });
        }
    }

    // --- constraint feasibility ---
    let mut g = vec![0.0; m];
    if !tnlp.eval_g(&x, true, &mut g) {
        return Err("eval_g failed at the claimed solution".to_string());
    }
    let mut max_con_violation = 0.0_f64;
    let mut worst_con: Option<RowReport> = None;
    for i in 0..m {
        let viol = box_violation(g[i], g_l[i], g_u[i]);
        if viol > max_con_violation {
            max_con_violation = viol;
            worst_con = Some(RowReport {
                index: i,
                name: name_at(&con_names, i, 'c'),
                value: g[i],
                lo: g_l[i],
                hi: g_u[i],
                violation: viol,
            });
        }
    }

    let feasible = max_con_violation <= args.feas_tol && max_bound_violation <= args.feas_tol;

    // --- objective ---
    let objective = tnlp.eval_f(&x, true);

    // --- first-order / KKT stationarity (only when duals are supplied) ---
    let mut stationarity = None;
    let mut dual_sign = None;
    let mut complementarity = None;
    let mut optimal = None;
    if duals_present {
        let lambda = &parsed.lambda;

        // ∇f(x*)
        let mut grad_f = vec![0.0; n];
        tnlp.eval_grad_f(&x, true, &mut grad_f);

        // Jacobian triplets (structure then values).
        let mut irow = vec![0i32; nnz];
        let mut jcol = vec![0i32; nnz];
        tnlp.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        );
        let mut jval = vec![0.0; nnz];
        tnlp.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Values { values: &mut jval },
        );

        // AMPL's dual sign convention can flip relative to ours; rather
        // than guess, compute the bound-projected stationarity residual
        // for both signs and keep the better one. A genuine KKT point is
        // stationary for exactly one of them; we report which.
        let (resid_pos, _comp_pos) = stationarity_residual(
            1.0, &grad_f, &irow, &jcol, &jval, fortran, lambda, &x, &x_l, &x_u,
        );
        let (resid_neg, _comp_neg) = stationarity_residual(
            -1.0, &grad_f, &irow, &jcol, &jval, fortran, lambda, &x, &x_l, &x_u,
        );
        let (best_resid, sign) = if resid_pos <= resid_neg {
            (resid_pos, 1)
        } else {
            (resid_neg, -1)
        };
        stationarity = Some(best_resid);
        dual_sign = Some(sign);
        complementarity = Some(constraint_complementarity(lambda, &g, &g_l, &g_u));
        optimal = Some(best_resid <= args.opt_tol);
    }

    // Verified = feasible (always required) AND, if --require-optimal,
    // also first-order optimal.
    let verified = feasible && (!args.require_optimal || optimal.unwrap_or(false));

    Ok(VerifyOutcome {
        n_vars: n,
        n_cons: m,
        nl_sha256,
        sol_sha256,
        solve_result_num: parsed.solve_result_num,
        feas_tol: args.feas_tol,
        opt_tol: args.opt_tol,
        max_con_violation,
        worst_con,
        max_bound_violation,
        worst_bound,
        feasible,
        objective,
        duals_present,
        stationarity,
        dual_sign,
        complementarity,
        optimal,
        verified,
    })
}

/// Bound-projected stationarity (a.k.a. "dual infeasibility"):
/// `s = ∇f + sign·Jᵀλ`, then for each variable the part of `s` that a
/// valid sign-constrained bound multiplier `z_L, z_U ≥ 0` cannot absorb.
/// Returns `(‖projected s‖∞, _)`.
#[allow(clippy::too_many_arguments)]
fn stationarity_residual(
    sign: Number,
    grad_f: &[Number],
    irow: &[i32],
    jcol: &[i32],
    jval: &[Number],
    fortran: bool,
    lambda: &[Number],
    x: &[Number],
    x_l: &[Number],
    x_u: &[Number],
) -> (Number, Number) {
    let n = grad_f.len();
    let off = if fortran { 1 } else { 0 };
    let mut s = grad_f.to_vec();
    for k in 0..jval.len() {
        let row = (irow[k] as usize).wrapping_sub(off);
        let col = (jcol[k] as usize).wrapping_sub(off);
        if row < lambda.len() && col < n {
            s[col] += sign * jval[k] * lambda[row];
        }
    }
    // Activity tolerance for "x_j sits on a bound."
    let mut dual_inf = 0.0_f64;
    for j in 0..n {
        let at_lo = is_finite_bound(x_l[j]) && (x[j] - x_l[j]).abs() <= 1e-8 * (1.0 + x_l[j].abs());
        let at_hi = is_finite_bound(x_u[j]) && (x_u[j] - x[j]).abs() <= 1e-8 * (1.0 + x_u[j].abs());
        let fixed =
            is_finite_bound(x_l[j]) && is_finite_bound(x_u[j]) && (x_u[j] - x_l[j]).abs() <= 1e-12;
        let r = if fixed {
            0.0
        } else if at_lo && !at_hi {
            // need z_L = s_j ≥ 0; leftover is the negative part.
            (-s[j]).max(0.0)
        } else if at_hi && !at_lo {
            // need z_U = -s_j ≥ 0; leftover is the positive part.
            s[j].max(0.0)
        } else {
            s[j].abs()
        };
        dual_inf = dual_inf.max(r);
    }
    (dual_inf, 0.0)
}

/// `max_i |λ_i| · dist(g_i, active side)` over constraints with a finite
/// range — a constraint with a nonzero multiplier should be active.
/// Equalities (`g_l == g_u`) contribute 0. Best-effort, informational.
fn constraint_complementarity(
    lambda: &[Number],
    g: &[Number],
    g_l: &[Number],
    g_u: &[Number],
) -> Number {
    let mut comp = 0.0_f64;
    for i in 0..lambda.len() {
        if (g_u[i] - g_l[i]).abs() <= 1e-12 {
            continue; // equality: multiplier is free, no complementarity
        }
        let dl = if is_finite_bound(g_l[i]) {
            (g[i] - g_l[i]).abs()
        } else {
            Number::INFINITY
        };
        let du = if is_finite_bound(g_u[i]) {
            (g_u[i] - g[i]).abs()
        } else {
            Number::INFINITY
        };
        let dist = dl.min(du);
        if dist.is_finite() {
            comp = comp.max(lambda[i].abs() * dist);
        }
    }
    comp
}

fn name_at(names: &[String], i: usize, kind: char) -> String {
    match names.get(i) {
        Some(s) if !s.is_empty() => s.clone(),
        _ => format!("{kind}[{i}]"),
    }
}

// ---------------------------------------------------------------------------
// AMPL .sol parser (the inverse of `crate::nl_writer`).
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ParsedSol {
    x: Vec<Number>,
    lambda: Vec<Number>,
    solve_result_num: Option<i32>,
}

/// Parse the ASCII AMPL `.sol` form pounce writes: a free-text banner, a
/// blank line, `Options`, an option count + that many option words, the
/// four-integer count block `<n_dual> <m> <n_primal> <n>`, then the dual
/// block followed by the primal block, then an optional `objno` line.
fn parse_sol(text: &str) -> Result<ParsedSol, String> {
    // Find the "Options" delimiter line, then tokenize everything after it.
    let mut after_options = None;
    for (i, line) in text.lines().enumerate() {
        if line.trim() == "Options" {
            after_options = Some(i);
            break;
        }
    }
    let start = after_options.ok_or("malformed .sol: no `Options` section found")?;
    let tail: String = text.lines().skip(start + 1).collect::<Vec<_>>().join(" ");
    let mut toks = tail.split_whitespace();

    let nopts: usize = toks
        .next()
        .ok_or("malformed .sol: missing option count")?
        .parse()
        .map_err(|e| format!("malformed .sol: bad option count: {e}"))?;
    for _ in 0..nopts {
        toks.next()
            .ok_or("malformed .sol: truncated option words")?;
    }

    let next_usize = |toks: &mut std::str::SplitWhitespace, what: &str| -> Result<usize, String> {
        toks.next()
            .ok_or_else(|| format!("malformed .sol: missing {what}"))?
            .parse::<usize>()
            .map_err(|e| format!("malformed .sol: bad {what}: {e}"))
    };
    let n_dual = next_usize(&mut toks, "dual count")?;
    let _m = next_usize(&mut toks, "constraint count")?;
    let n_primal = next_usize(&mut toks, "primal count")?;
    let _n = next_usize(&mut toks, "variable count")?;

    let mut lambda = Vec::with_capacity(n_dual);
    for k in 0..n_dual {
        let t = toks
            .next()
            .ok_or_else(|| format!("malformed .sol: truncated dual block at {k}"))?;
        lambda.push(
            t.parse::<Number>()
                .map_err(|e| format!("malformed .sol: bad dual {k}: {e}"))?,
        );
    }
    let mut x = Vec::with_capacity(n_primal);
    for k in 0..n_primal {
        let t = toks
            .next()
            .ok_or_else(|| format!("malformed .sol: truncated primal block at {k}"))?;
        x.push(
            t.parse::<Number>()
                .map_err(|e| format!("malformed .sol: bad primal {k}: {e}"))?,
        );
    }

    // Optional `objno <objno> <solve_result_num>`.
    let mut solve_result_num = None;
    let rest: Vec<&str> = toks.collect();
    if let Some(p) = rest.iter().position(|&t| t == "objno") {
        if let Some(code) = rest.get(p + 2) {
            solve_result_num = code.parse::<i32>().ok();
        }
    }

    Ok(ParsedSol {
        x,
        lambda,
        solve_result_num,
    })
}

// ---------------------------------------------------------------------------
// Console + JSON rendering.
// ---------------------------------------------------------------------------

fn print_report(args: &VerifyArgs, o: &VerifyOutcome) {
    println!("pounce verify — independent solution check");
    println!(
        "  problem : {}  ({} vars, {} cons)",
        args.nl.display(),
        o.n_vars,
        o.n_cons
    );
    println!("            sha256:{}", o.nl_sha256);
    println!("  solution: {}", args.sol.display());
    println!("            sha256:{}", o.sol_sha256);
    if let Some(srn) = o.solve_result_num {
        println!("  claimed solve_result_num: {srn}");
    }
    println!();
    println!("  feasibility (tol {:.1e}):", o.feas_tol);
    print_row(
        "max constraint violation",
        o.max_con_violation,
        &o.worst_con,
    );
    print_row(
        "max bound violation     ",
        o.max_bound_violation,
        &o.worst_bound,
    );
    if let Some(obj) = o.objective {
        println!("  objective at x*: {obj:.10e}");
    }
    if o.duals_present {
        println!();
        println!("  optimality (tol {:.1e}, duals supplied):", o.opt_tol);
        if let Some(s) = o.stationarity {
            let sign = o.dual_sign.unwrap_or(1);
            println!("    KKT stationarity residual: {s:.3e}  (dual sign {sign:+})");
        }
        if let Some(c) = o.complementarity {
            println!("    complementarity residual : {c:.3e}");
        }
    } else {
        println!();
        println!("  optimality: not checked (.sol carried no duals)");
    }
    println!();
    let verdict = if o.verified {
        "VERIFIED — solution is feasible for the canonical problem".to_string()
    } else if !o.feasible {
        "REJECTED — solution VIOLATES the canonical constraints".to_string()
    } else {
        "REJECTED — feasible but not first-order optimal (--require-optimal)".to_string()
    };
    println!("  VERDICT: {verdict}");
}

fn print_row(label: &str, v: Number, worst: &Option<RowReport>) {
    match worst {
        Some(r) => println!(
            "    {label}: {v:.3e}  at {} (value {:.6e}, bounds [{:.6e}, {:.6e}])",
            r.name, r.value, r.lo, r.hi
        ),
        None => println!("    {label}: {v:.3e}"),
    }
}

/// Environment variable holding the HMAC key. When set (non-empty) and a
/// `--json-output` receipt is requested, the receipt is signed.
pub const KEY_ENV: &str = "POUNCE_VERIFY_KEY";

/// The exact byte string that gets HMAC-signed. Deliberately **float-free**
/// — only hex hashes, integer counts, and the verdict — so any language
/// reproduces it byte-for-byte (no float-formatting parity problems between
/// Rust and a Python/JS consumer). One `key=value` per line, fixed order,
/// trailing newline. The consumer re-derives this from the receipt fields,
/// recomputes `HMAC-SHA256(key, preimage)`, and compares to `signature`.
/// Documented in `docs/src/verify.md`.
///
/// The signed fields are exactly the security-critical bindings: *which*
/// problem (`nl_sha256`), *which* solution (`sol_sha256`), the problem
/// dimensions, and the verdict. The numeric violations in the receipt are
/// supporting evidence; trust flows from the hashes + `verified` flag.
pub fn signing_preimage(o: &VerifyOutcome) -> String {
    format!(
        "pounce-verify-receipt/v1\n\
         verify_version=1\n\
         nl_sha256={}\n\
         sol_sha256={}\n\
         n_vars={}\n\
         n_cons={}\n\
         feasible={}\n\
         verified={}\n\
         verdict={}\n",
        o.nl_sha256,
        o.sol_sha256,
        o.n_vars,
        o.n_cons,
        o.feasible,
        o.verified,
        if o.verified { "VERIFIED" } else { "REJECTED" },
    )
}

fn receipt_json(args: &VerifyArgs, o: &VerifyOutcome) -> String {
    use serde_json::json;
    let worst_con = o.worst_con.as_ref().map(row_json);
    let worst_bound = o.worst_bound.as_ref().map(row_json);
    let optimality = if o.duals_present {
        json!({
            "available": true,
            "objective": o.objective,
            "stationarity_residual": o.stationarity,
            "dual_sign": o.dual_sign,
            "complementarity_residual": o.complementarity,
            "optimal": o.optimal,
            "note": "bound-projected stationarity (dual infeasibility) using the .sol's \
                     constraint duals; bound multipliers inferred from activity. Sign chosen \
                     to match the supplied dual convention. Feasibility is the rigorous gate."
        })
    } else {
        json!({ "available": false })
    };
    let mut receipt = json!({
        "pounce_verify_version": 1,
        "solver": format!("pounce {}", env!("CARGO_PKG_VERSION")),
        "problem": {
            "path": args.nl.display().to_string(),
            "sha256": o.nl_sha256,
            "n_vars": o.n_vars,
            "n_cons": o.n_cons,
        },
        "solution": {
            "path": args.sol.display().to_string(),
            "sha256": o.sol_sha256,
            "claimed_solve_result_num": o.solve_result_num,
            "duals_present": o.duals_present,
        },
        "tolerances": { "feasibility": o.feas_tol, "optimality": o.opt_tol },
        "feasibility": {
            "max_constraint_violation": o.max_con_violation,
            "worst_constraint": worst_con,
            "max_bound_violation": o.max_bound_violation,
            "worst_bound": worst_bound,
            "feasible": o.feasible,
        },
        "optimality": optimality,
        "verdict": if o.verified { "VERIFIED" } else { "REJECTED" },
        "verified": o.verified,
    });

    // Sign the receipt when a key is present. The signature covers the
    // float-free `signing_preimage`, NOT the pretty JSON, so a consumer in
    // any language can recompute it without matching float formatting.
    if let Ok(key) = std::env::var(KEY_ENV) {
        if !key.is_empty() {
            if let Some(obj) = receipt.as_object_mut() {
                let sig = sha256::hmac_hex(key.as_bytes(), signing_preimage(o).as_bytes());
                obj.insert("signature_alg".into(), json!("HMAC-SHA256"));
                obj.insert(
                    "signed_fields".into(),
                    json!([
                        "verify_version",
                        "nl_sha256",
                        "sol_sha256",
                        "n_vars",
                        "n_cons",
                        "feasible",
                        "verified",
                        "verdict"
                    ]),
                );
                obj.insert("signature".into(), json!(sig));
            }
        }
    }

    serde_json::to_string_pretty(&receipt).unwrap_or_else(|_| "{}".to_string())
}

fn row_json(r: &RowReport) -> serde_json::Value {
    serde_json::json!({
        "index": r.index,
        "name": r.name,
        "value": r.value,
        "lower": r.lo,
        "upper": r.hi,
        "violation": r.violation,
    })
}

// ---------------------------------------------------------------------------
// Self-contained SHA-256 (FIPS 180-4) — content-addresses the receipt's
// inputs with zero new dependencies, matching the crate's hand-rolled,
// dependency-light style. Known-answer tested below.
// ---------------------------------------------------------------------------

pub mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    /// Raw 32-byte SHA-256 digest.
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        // Pad: message || 0x80 || 0x00... || 64-bit big-endian bit length.
        let bit_len = (data.len() as u64).wrapping_mul(8);
        let mut msg = data.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());

        let mut w = [0u32; 64];
        for chunk in msg.chunks_exact(64) {
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    chunk[4 * i],
                    chunk[4 * i + 1],
                    chunk[4 * i + 2],
                    chunk[4 * i + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
                (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn to_hex(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }

    /// Lowercase-hex SHA-256 of `data`.
    pub fn hex(data: &[u8]) -> String {
        to_hex(&digest(data))
    }

    /// HMAC-SHA256(key, msg) per RFC 2104, raw 32 bytes.
    pub fn hmac(key: &[u8], msg: &[u8]) -> [u8; 32] {
        const BLOCK: usize = 64;
        let mut k = [0u8; BLOCK];
        if key.len() > BLOCK {
            k[..32].copy_from_slice(&digest(key));
        } else {
            k[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0x36u8; BLOCK];
        let mut opad = [0x5cu8; BLOCK];
        for i in 0..BLOCK {
            ipad[i] ^= k[i];
            opad[i] ^= k[i];
        }
        let mut inner = Vec::with_capacity(BLOCK + msg.len());
        inner.extend_from_slice(&ipad);
        inner.extend_from_slice(msg);
        let inner_digest = digest(&inner);
        let mut outer = Vec::with_capacity(BLOCK + 32);
        outer.extend_from_slice(&opad);
        outer.extend_from_slice(&inner_digest);
        digest(&outer)
    }

    /// HMAC-SHA256 as lowercase hex.
    pub fn hmac_hex(key: &[u8], msg: &[u8]) -> String {
        to_hex(&hmac(key, msg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_writer::{format_sol, SolutionFile};

    #[test]
    fn sha256_known_answers() {
        // FIPS 180-4 test vectors.
        assert_eq!(
            sha256::hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256::hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256::hex(b"The quick brown fox jumps over the lazy dog"),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
    }

    #[test]
    fn hmac_sha256_known_answers() {
        // RFC 4231 test case 2.
        assert_eq!(
            sha256::hmac_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // RFC 4231 test case 1: key = 0x0b * 20, data = "Hi There".
        assert_eq!(
            sha256::hmac_hex(&[0x0b; 20], b"Hi There"),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn parse_sol_round_trips_writer() {
        // Writer is the inverse we must match exactly.
        let payload = SolutionFile {
            message: "POUNCE 0.3.1: Optimal Solution Found",
            x: &[1.0, 2.5, -0.5, 100.0],
            lambda: &[0.1, -0.2],
            solve_result_num: 0,
            suffixes: &[],
        };
        let text = format_sol(&payload);
        let parsed = parse_sol(&text).expect("parse");
        assert_eq!(parsed.x.len(), 4);
        assert_eq!(parsed.lambda.len(), 2);
        assert!((parsed.x[1] - 2.5).abs() < 1e-15);
        assert!((parsed.x[3] - 100.0).abs() < 1e-12);
        assert!((parsed.lambda[0] - 0.1).abs() < 1e-15);
        assert_eq!(parsed.solve_result_num, Some(0));
    }

    #[test]
    fn parse_sol_handles_no_duals() {
        let payload = SolutionFile {
            message: "msg",
            x: &[3.0, 4.0],
            lambda: &[],
            solve_result_num: 200,
            suffixes: &[],
        };
        let text = format_sol(&payload);
        let parsed = parse_sol(&text).expect("parse");
        assert_eq!(parsed.x, vec![3.0, 4.0]);
        assert!(parsed.lambda.is_empty());
        assert_eq!(parsed.solve_result_num, Some(200));
    }

    #[test]
    fn box_violation_basic() {
        // inside
        assert_eq!(box_violation(5.0, 0.0, 10.0), 0.0);
        // below lower
        assert!((box_violation(-2.0, 0.0, 10.0) - 2.0).abs() < 1e-15);
        // above upper
        assert!((box_violation(13.0, 0.0, 10.0) - 3.0).abs() < 1e-15);
        // one-sided (no upper)
        assert_eq!(box_violation(1e30, 0.0, NLP_UPPER_BOUND_INF), 0.0);
    }
}
