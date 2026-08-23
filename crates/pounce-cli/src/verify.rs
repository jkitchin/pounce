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
//!
//! # Two different complementarity quantities (gh #516)
//!
//! "Complementarity" names two distinct residuals, and printing either one
//! under the bare label invites a comparison against the other:
//!
//! * **constraint** complementarity — `max_i |λ_i| · dist(g_i, nearest
//!   finite side)` over **rows**, from the `.sol`'s constraint duals. This
//!   is the one `verify` has always computed.
//! * **bound** complementarity — `max_j max(|z_L·(x−x_L)|, |z_U·(x_U−x)|)`
//!   over **variables**, from the bound multipliers. This is what Ipopt
//!   prints as `Complementarity`, and it needs the `ipopt_zL_out` /
//!   `ipopt_zU_out` `.sol` suffixes.
//!
//! They can differ by many orders of magnitude at the same point and
//! neither is wrong for what it measures. `verify` therefore names both
//! explicitly, reads the bound multipliers when the `.sol` carries them,
//! and says "not checked" — rather than nothing — when it does not.
//!
//! Those same suffixes also sharpen stationarity: without them the residual
//! is bound-*projected* and cannot see a bound multiplier that is missing or
//! wrong (gh #495); with them the exact residual is available, and
//! `--require-optimal` gates on it.

use crate::nl_reader;
use pounce_common::types::Number;
use pounce_nlp::diagnostics::RowReport;
use pounce_nlp::diagnostics::verify::{
    SolutionClaim, VerifyOptions, VerifyOutcome, VerifyProvenance, verify_tnlp,
};
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

Complementarity: two different residuals carry that name, and they can
differ by many orders of magnitude at the same point.
  * constraint complementarity (rows, |lambda|*slack) is computed from the
    .sol's constraint duals and is always reported alongside stationarity.
  * bound complementarity (vars, |z|*slack) is the quantity Ipopt prints as
    `Complementarity`. It needs the bound multipliers, which reach a .sol
    only as the `ipopt_zL_out` / `ipopt_zU_out` suffixes; without them it is
    reported as `not checked`, never as a number.
Do not compare the row quantity against a solver's `Complementarity` line.

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
    let provenance_hashes = (sha256::hex(&nl_bytes), sha256::hex(&sol_bytes));

    // --- canonical problem ---
    let prob = nl_reader::read_nl_file(&args.nl)?;
    let n = prob.n;
    let con_names = prob.con_names.clone();
    let var_names = prob.var_names.clone();
    let mut tnlp = nl_reader::NlTnlp::new(prob);

    // --- claimed solution ---
    let sol_text = String::from_utf8_lossy(&sol_bytes);
    let parsed = parse_sol(&sol_text)?;
    let claim = SolutionClaim {
        x: parsed.x,
        lambda: parsed.lambda,
        z_l: parsed.z_l,
        z_u: parsed.z_u,
    };
    let provenance = VerifyProvenance {
        nl_sha256: provenance_hashes.0,
        sol_sha256: provenance_hashes.1,
        solve_result_num: parsed.solve_result_num,
    };
    let opts = VerifyOptions {
        feas_tol: args.feas_tol,
        opt_tol: args.opt_tol,
        require_optimal: args.require_optimal,
    };

    // `n` is the `.nl`'s variable count; the core re-derives it from
    // `get_nlp_info` and reports the mismatch, so this is only here to keep
    // the message naming the two *files* the user passed.
    if claim.x.len() != n {
        return Err(format!(
            "solution has {} primal values but the problem has {n} variables \
             (is this the right .sol for this .nl?)",
            claim.x.len()
        ));
    }

    verify_tnlp(
        &mut tnlp,
        &claim,
        &var_names,
        &con_names,
        &provenance,
        &opts,
    )
}

// ---------------------------------------------------------------------------
// AMPL .sol parser (the inverse of `crate::nl_writer`).
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ParsedSol {
    x: Vec<Number>,
    lambda: Vec<Number>,
    solve_result_num: Option<i32>,
    /// `ipopt_zL_out` variable suffix, densified to `n`, when present.
    z_l: Option<Vec<Number>>,
    /// `ipopt_zU_out` variable suffix, densified to `n`, when present.
    z_u: Option<Vec<Number>>,
}

/// Parse the ASCII AMPL `.sol` form pounce writes: a free-text banner, a
/// blank line, `Options`, an option count + that many option words, the
/// four-integer count block `<n_dual> <m> <n_primal> <n>`, then the dual
/// block followed by the primal block, then an optional `objno` line and any
/// number of suffix blocks.
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

    // Trailing section: an optional `objno <objno> <solve_result_num>` and
    // any number of suffix blocks.
    let rest: Vec<&str> = toks.collect();
    let (solve_result_num, var_suffixes) = parse_sol_tail(&rest, n_primal);
    let suffix = |name: &str| -> Option<Vec<Number>> {
        var_suffixes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
    };

    Ok(ParsedSol {
        x,
        lambda,
        solve_result_num,
        z_l: suffix("ipopt_zL_out"),
        z_u: suffix("ipopt_zU_out"),
    })
}

/// Walk the tokens after the primal block: an optional
/// `objno <objno> <solve_result_num>` and any number of suffix blocks, each
/// `suffix <kind> <nvalues> <namelen> <tablen> <tabline>`, the name on its
/// own line, then `<idx> <value>` pairs (see `pounce_nl::sol_writer` for the
/// shape pounce writes and Ipopt's AMPL interface writes back).
///
/// Returns the `solve_result_num` and every **variable-indexed real**
/// suffix, densified to `n` — a `.sol` sparse-trims zero entries, so an
/// absent index means zero, not missing.
///
/// A malformed or unsupported block stops the walk and keeps what was read
/// so far: a `.sol` is still perfectly usable for the feasibility check that
/// is this tool's actual gate, and a parse error there must not turn a
/// checkable solution into an I/O failure.
fn parse_sol_tail(rest: &[&str], n: usize) -> (Option<i32>, Vec<(String, Vec<Number>)>) {
    let mut solve_result_num = None;
    let mut out: Vec<(String, Vec<Number>)> = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        match rest[i] {
            "objno" => {
                solve_result_num = rest.get(i + 2).and_then(|t| t.parse::<i32>().ok());
                i += 3;
            }
            "suffix" => {
                let int_at = |k: usize| rest.get(i + k).and_then(|t| t.parse::<i64>().ok());
                let (Some(kind), Some(nvalues), Some(tablen)) = (int_at(1), int_at(2), int_at(4))
                else {
                    break;
                };
                let (Some(name), true) = (rest.get(i + 6), nvalues >= 0) else {
                    break;
                };
                let name = (*name).to_string();
                i += 7;
                // A suffix value table follows the name as free text we
                // cannot delimit by whitespace, so its tokens would be
                // mis-read as values. Neither pounce nor Ipopt writes one.
                if tablen != 0 {
                    break;
                }
                // Low two bits pick the target (0 = var), 0x4 flags a real
                // payload — ASL's `ASL_Sufkind_*` bits.
                let want = (kind & 0x3) == 0 && (kind & 0x4) != 0;
                let mut dense = vec![0.0; n];
                let mut complete = true;
                for _ in 0..nvalues as usize {
                    let (Some(it), Some(vt)) = (rest.get(i), rest.get(i + 1)) else {
                        complete = false;
                        break;
                    };
                    if let (true, Ok(idx), Ok(v)) =
                        (want, it.parse::<usize>(), vt.parse::<Number>())
                        && idx < n
                    {
                        dense[idx] = v;
                    }
                    i += 2;
                }
                if !complete {
                    break;
                }
                if want {
                    out.push((name, dense));
                }
            }
            _ => i += 1,
        }
    }
    (solve_result_num, out)
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
    if o.stationarity.is_some() || o.bound_multipliers_present {
        let source = match (o.duals_present, o.bound_multipliers_present) {
            (true, true) => "duals + bound multipliers supplied",
            (true, false) => "duals supplied",
            (false, true) => "bound multipliers supplied",
            (false, false) => "no rows, so no duals to supply",
        };
        println!();
        println!("  optimality (tol {:.1e}, {source}):", o.opt_tol);
        if let Some(s) = o.stationarity {
            let sign = o.dual_sign.unwrap_or(1);
            println!(
                "    KKT stationarity residual (bound-projected)  : {s:.3e}  (dual sign {sign:+})"
            );
        }
        if let Some(s) = o.stationarity_with_bound_multipliers {
            println!("    dual infeasibility (with z_L/z_U suffixes)   : {s:.3e}");
        }
        // Two different residuals answer to "complementarity", and the row
        // one is NOT what a solver prints as `Complementarity` — label both
        // by what they range over so the numbers cannot be crossed (gh #516).
        if let Some(c) = o.constraint_complementarity {
            println!("    constraint complementarity (rows, |λ|·slack) : {c:.3e}");
        }
        match o.bound_complementarity {
            Some(c) => println!("    bound complementarity (vars, |z|·slack)      : {c:.3e}"),
            None => {
                println!(
                    "    bound complementarity (vars, |z|·slack)      : not checked \
                     — the .sol carries no"
                );
                println!(
                    "      ipopt_zL_out/ipopt_zU_out suffixes. This, not the row line \
                     above, is the"
                );
                println!("      quantity a solver reports as `Complementarity`.");
            }
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
    } else if o.optimal.is_none() {
        // Feasible, --require-optimal was asked for, but optimality could not
        // be checked at all because the .sol carried no duals — say so rather
        // than implying we found it non-optimal.
        "REJECTED — feasible, but --require-optimal needs duals and the .sol \
         carried none"
            .to_string()
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
    let optimality = if o.duals_present || o.bound_multipliers_present {
        // Optimality is a property of a FEASIBLE point, so this must not report
        // `true` for one that violates the constraints. The stationarity
        // residual of an infeasible point can be legitimately zero, which
        // previously surfaced as `optimality.optimal: true` inside a receipt
        // whose verdict was REJECTED — the top-level fields were correct, but a
        // consumer reading this nested field alone was told the opposite.
        // The raw residuals are still reported unconditioned: they are useful
        // for diagnosing *why* a point failed.
        let optimal = o.optimal.map(|opt| opt && o.feasible);
        json!({
            "available": true,
            "objective": o.objective,
            "stationarity_residual": o.stationarity,
            "dual_sign": o.dual_sign,
            "stationarity_residual_with_bound_multipliers":
                o.stationarity_with_bound_multipliers,
            "constraint_complementarity_residual": o.constraint_complementarity,
            "bound_complementarity_residual": o.bound_complementarity,
            "bound_multipliers_present": o.bound_multipliers_present,
            // Deprecated alias, kept so a v1 consumer does not break. Its bare
            // name is the trap gh #516 is about: read
            // `constraint_complementarity_residual` instead.
            "complementarity_residual": o.constraint_complementarity,
            "optimal": optimal,
            "note": "`stationarity_residual` is the BOUND-PROJECTED dual infeasibility from \
                     the .sol's constraint duals, with bound multipliers inferred from \
                     activity; the sign is chosen to match the supplied dual convention. \
                     `constraint_complementarity_residual` is max_i |lambda_i| * dist(g_i, \
                     nearest finite side) over ROWS — it is NOT what a solver reports as \
                     `Complementarity`. That is `bound_complementarity_residual`, \
                     max_j max(|z_L*(x-x_L)|, |z_U*(x_U-x)|) over VARIABLES, available only \
                     when the .sol carries the ipopt_zL_out/ipopt_zU_out suffixes (null \
                     otherwise, meaning not checked — not zero). When those suffixes are \
                     present, `stationarity_residual_with_bound_multipliers` is the exact, \
                     unprojected residual and is what `--require-optimal` gates on. \
                     `complementarity_residual` is a deprecated alias of \
                     `constraint_complementarity_residual`. Feasibility is the rigorous \
                     gate, and `optimal` is reported false for an infeasible point \
                     regardless of its stationarity residual."
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
    use crate::nl_writer::{SolutionFile, format_sol};
    use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};

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
        // Writer is the inverse we must match exactly. Derive the banner
        // from the crate version so this fixture never goes stale on a
        // version bump (the round-trip is agnostic to the exact string).
        let message = format!(
            "POUNCE {}: Optimal Solution Found",
            env!("CARGO_PKG_VERSION")
        );
        let payload = SolutionFile {
            message: &message,
            x: &[1.0, 2.5, -0.5, 100.0],
            mult_g: &[0.1, -0.2],
            solve_result_num: 0,
            suffixes: &[],
        };
        let text = format_sol(&payload);
        let parsed = parse_sol(&text).expect("parse");
        assert_eq!(parsed.x.len(), 4);
        assert_eq!(parsed.lambda.len(), 2);
        assert!((parsed.x[1] - 2.5).abs() < 1e-15);
        assert!((parsed.x[3] - 100.0).abs() < 1e-12);
        // The primal round-trips as an identity, but the dual block does
        // NOT: `format_sol` negates pounce's internal multipliers into the
        // AMPL marginal convention (gh #271), and `parse_sol` reads the
        // file back verbatim. So a `mult_g` of +0.1 must come back as a
        // parsed dual of -0.1. Asserting identity here is what previously
        // let the sign defect pass unnoticed.
        assert!((parsed.lambda[0] + 0.1).abs() < 1e-15);
        assert!((parsed.lambda[1] - 0.2).abs() < 1e-15);
        assert_eq!(parsed.solve_result_num, Some(0));
    }

    #[test]
    fn parse_sol_handles_no_duals() {
        let payload = SolutionFile {
            message: "msg",
            x: &[3.0, 4.0],
            mult_g: &[],
            solve_result_num: 200,
            suffixes: &[],
        };
        let text = format_sol(&payload);
        let parsed = parse_sol(&text).expect("parse");
        assert_eq!(parsed.x, vec![3.0, 4.0]);
        assert!(parsed.lambda.is_empty());
        assert_eq!(parsed.solve_result_num, Some(200));
    }

    // -----------------------------------------------------------------
    // gh #516 — the two complementarity quantities.
    // -----------------------------------------------------------------

    /// The bound multipliers reach a `.sol` only as suffix blocks, so the
    /// parser has to pick them out of the trailing section — past `objno`
    /// and past whatever other suffixes the writer emitted.
    #[test]
    fn parse_sol_reads_the_bound_multiplier_suffixes() {
        use crate::nl_writer::{SolSuffix, SolSuffixTarget, SolSuffixValues};
        let payload = SolutionFile {
            message: "msg",
            x: &[1.0, -1.0, 0.0],
            mult_g: &[0.5],
            solve_result_num: 0,
            suffixes: &[
                // An unrelated block first: the walk must step over it.
                SolSuffix {
                    name: "sens_sol_state_1".to_string(),
                    target: SolSuffixTarget::Var,
                    values: SolSuffixValues::Real(vec![9.0, 9.0, 9.0]),
                },
                SolSuffix {
                    name: "ipopt_zL_out".to_string(),
                    target: SolSuffixTarget::Var,
                    values: SolSuffixValues::Real(vec![0.0, 2.0, 0.0]),
                },
                SolSuffix {
                    name: "ipopt_zU_out".to_string(),
                    target: SolSuffixTarget::Var,
                    values: SolSuffixValues::Real(vec![-4.0, 0.0, 0.0]),
                },
            ],
        };
        let parsed = parse_sol(&format_sol(&payload)).expect("parse");
        assert_eq!(parsed.solve_result_num, Some(0), "objno still parses");
        // Densified back to `n`: the writer sparse-trims zeros, so an absent
        // index means zero — not a short vector, and not "missing".
        assert_eq!(parsed.z_l, Some(vec![0.0, 2.0, 0.0]));
        assert_eq!(parsed.z_u, Some(vec![-4.0, 0.0, 0.0]));
    }

    /// No suffixes → bound complementarity is *not checked*, and must stay
    /// `None` rather than collapse to a comfortable `0.0`.
    #[test]
    fn parse_sol_reports_absent_bound_multipliers_as_absent() {
        let payload = SolutionFile {
            message: "msg",
            x: &[1.0],
            mult_g: &[0.5],
            solve_result_num: 0,
            suffixes: &[],
        };
        let parsed = parse_sol(&format_sol(&payload)).expect("parse");
        assert!(parsed.z_l.is_none() && parsed.z_u.is_none());
    }
}
