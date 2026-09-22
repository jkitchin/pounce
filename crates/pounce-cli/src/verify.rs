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
use pounce_common::tolerance::is_negligible;
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};
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
    /// `max_i |λ_i| · dist(g_i, active side)` over **rows**. NOT the
    /// quantity a solver reports as `Complementarity` — see
    /// [`bound_complementarity`](VerifyOutcome::bound_complementarity).
    pub constraint_complementarity: Option<Number>,
    /// Whether the `.sol` carried `ipopt_zL_out` / `ipopt_zU_out`.
    pub bound_multipliers_present: bool,
    /// `max_j max(|z_L·(x−x_L)|, |z_U·(x_U−x)|)` over **variables** — the
    /// quantity Ipopt prints as `Complementarity`. `None` when the `.sol`
    /// carried no bound multipliers, in which case it is *not checked*
    /// rather than zero.
    pub bound_complementarity: Option<Number>,
    /// Exact (non-projected) dual infeasibility
    /// `‖∇f + sign·Jᵀλ − (z_L^suffix + z_U^suffix)‖∞`, available only when
    /// both duals and bound multipliers are present.
    pub stationarity_with_bound_multipliers: Option<Number>,
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

// `is_finite_bound(b) = b > NLP_LOWER_BOUND_INF && b < NLP_UPPER_BOUND_INF`
// used to live here — a *band* membership test applied to lower and upper
// bounds alike (gh #403). A real upper bound of `-5e20` failed it, so
// `box_violation` scored `0.0` against it and `verify` reported ACCEPTED for a
// `.sol` that violates a declared bound. Presence is directional; use
// `lower_bound_present` / `upper_bound_present` from `pounce_common::types`,
// picking the one that matches the side you hold.

/// `g_l ≤ v ≤ g_u` violation: how far `v` is outside the box, 0 if inside.
///
/// A non-finite `v` (NaN or ±∞) is treated as an infinite violation, never
/// as feasible: `NaN`-laden arithmetic would otherwise collapse to `0.0`
/// through `f64::max` (which drops NaN operands) and let a fabricated `.sol`
/// slip past the feasibility gate — the exact threat this checker defends
/// against. An unbounded variable pinned at ±∞ is likewise not a real point.
/// The natural magnitude of a row, for a scale-relative feasibility test.
///
/// `verify` reads a `.nl` and a `.sol`; no solver scaling has been applied, so
/// the magnitude has to come from the row's own numbers — the evaluated value
/// and whichever bounds are finite. Infinite bounds carry no magnitude
/// information and are skipped.
pub(crate) fn row_magnitude(value: Number, lo: Number, hi: Number) -> Number {
    let mut m = if value.is_finite() { value.abs() } else { 0.0 };
    if lower_bound_present(lo) {
        m = m.max(lo.abs());
    }
    if upper_bound_present(hi) {
        m = m.max(hi.abs());
    }
    m
}

/// Whether a row's violation is real, judged relative to the row's own
/// magnitude.
///
/// An absolute tolerance is meaningless against a row evaluating near `1e13`:
/// `--feas-tol 1e-6` is unreachable there, so a solution correct to eleven
/// relative digits was reported REJECTED. Scaling the tolerance by the row
/// magnitude makes the verdict independent of how the model happens to be
/// written.
///
/// Uses the **accepting** direction (`is_negligible`), which is never stricter
/// than the plain absolute `tol`. A pure relative test was tried first and
/// rejected genuine solutions: the solver converges to *absolute* residuals, so
/// on a row of magnitude `1e-3` a residual of `1e-8` is converged, while a
/// relative test at `tol = 1e-6` would demand `1e-9`.
///
/// The non-finite case is handled here rather than inside the primitive, which
/// reports an unjudgeable value as not-negligible-and-not-significant. A `.sol`
/// carrying `NaN` or `±inf` is not a point at all and must be rejected — which
/// is what `box_violation` returning infinity encodes.
pub(crate) fn row_is_violated(viol: Number, magnitude: Number, feas_tol: Number) -> bool {
    if !viol.is_finite() {
        return true;
    }
    !is_negligible(viol, magnitude, feas_tol)
}

pub(crate) fn box_violation(v: Number, lo: Number, hi: Number) -> Number {
    if !v.is_finite() {
        return Number::INFINITY;
    }
    let below = if lower_bound_present(lo) {
        lo - v
    } else {
        Number::NEG_INFINITY
    };
    let above = if upper_bound_present(hi) {
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
    let mut any_bound_violated = false;
    for j in 0..n {
        let viol = box_violation(x[j], x_l[j], x_u[j]);
        if row_is_violated(viol, row_magnitude(x[j], x_l[j], x_u[j]), args.feas_tol) {
            any_bound_violated = true;
        }
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
    let mut any_con_violated = false;
    for i in 0..m {
        let viol = box_violation(g[i], g_l[i], g_u[i]);
        if row_is_violated(viol, row_magnitude(g[i], g_l[i], g_u[i]), args.feas_tol) {
            any_con_violated = true;
        }
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

    // Per-row and scale-relative: a single absolute threshold across rows of
    // wildly different magnitude answers a different question for each of them.
    let feasible = !any_con_violated && !any_bound_violated;

    // --- objective ---
    let objective = tnlp.eval_f(&x, true);

    // --- bound multipliers, when the `.sol` exported them (gh #516) ---
    //
    // The bound complementarity Ipopt prints as `Complementarity` cannot be
    // computed from the primal and the constraint duals alone; it needs
    // `z_L` / `z_U`, which reach a `.sol` only as the `ipopt_zL_out` /
    // `ipopt_zU_out` variable suffixes. Absent them the quantity is *not
    // checked* — never silently reported as the row quantity.
    let bound_multipliers_present = parsed.z_l.is_some() || parsed.z_u.is_some();
    let z_l_suf = parsed.z_l.clone().unwrap_or_else(|| vec![0.0; n]);
    let z_u_suf = parsed.z_u.clone().unwrap_or_else(|| vec![0.0; n]);
    let bound_complementarity = if bound_multipliers_present {
        Some(bound_complementarity(&x, &x_l, &x_u, &z_l_suf, &z_u_suf))
    } else {
        None
    };

    // --- first-order / KKT stationarity (only when duals are supplied) ---
    let mut stationarity = None;
    let mut dual_sign = None;
    let mut constraint_complementarity = None;
    let mut stationarity_with_bound_multipliers = None;
    let mut optimal = None;
    // A problem with no rows has no constraint duals to carry, so `∇f` alone
    // is the Lagrangian gradient and the residual is available from an empty
    // dual block — which is what a `.sol` for a bounds-only model has.
    if duals_present || m == 0 {
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
        let s_pos = lagrangian_gradient(1.0, &grad_f, &irow, &jcol, &jval, fortran, lambda);
        let s_neg = lagrangian_gradient(-1.0, &grad_f, &irow, &jcol, &jval, fortran, lambda);
        let resid_pos = bound_projected_residual(&s_pos, &x, &x_l, &x_u, args.opt_tol);
        let resid_neg = bound_projected_residual(&s_neg, &x, &x_l, &x_u, args.opt_tol);
        let (best_resid, sign, s) = if resid_pos <= resid_neg {
            (resid_pos, 1, &s_pos)
        } else {
            (resid_neg, -1, &s_neg)
        };
        stationarity = Some(best_resid);
        dual_sign = Some(sign);
        constraint_complementarity = Some(row_complementarity(lambda, &g, &g_l, &g_u));

        // With the bound multipliers in hand the residual no longer has to
        // be projected: the exact dual infeasibility is available, and it
        // is what a solver reports. It is also the strictly sharper check —
        // the projection can only *remove* residual — so `--require-optimal`
        // gates on it whenever it exists.
        if bound_multipliers_present {
            stationarity_with_bound_multipliers =
                Some(exact_dual_infeasibility(s, &z_l_suf, &z_u_suf));
        }
        let gate = stationarity_with_bound_multipliers.unwrap_or(best_resid);
        optimal = Some(gate <= args.opt_tol);
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
        constraint_complementarity,
        bound_multipliers_present,
        bound_complementarity,
        stationarity_with_bound_multipliers,
        optimal,
        verified,
    })
}

/// `s = ∇f + sign·Jᵀλ` — the part of the Lagrangian gradient the constraint
/// duals can account for, before any bound multiplier enters.
fn lagrangian_gradient(
    sign: Number,
    grad_f: &[Number],
    irow: &[i32],
    jcol: &[i32],
    jval: &[Number],
    fortran: bool,
    lambda: &[Number],
) -> Vec<Number> {
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
    s
}

/// Bound-**projected** stationarity (a.k.a. "dual infeasibility"): for each
/// variable, the part of `s` that a valid sign-constrained bound multiplier
/// `z_L, z_U ≥ 0` cannot absorb. Returns `‖projected s‖∞`.
///
/// Stationarity is `z_L − z_U = s`, so a positive `s_j` is absorbable by the
/// lower bound and a negative one by the upper. A bound may absorb it when
/// the multiplier that would take is complementary with the variable's gap
/// to that bound: `|s_j|·gap ≤ compl_tol`, or the gap is within `1e-8`
/// relative. The product is the test an interior-point answer passes. It
/// stops a barrier's distance from the bound, not on it, and that distance
/// is `μ/z`, so an activity test on the gap alone misreads its active bounds
/// as interior variables and reports their whole gradient as residual. On
/// PGLib case6468_rte, variables 1.9e-8 inside a bound of 0.2189 (under the
/// relative test's 1.2e-8) read as a residual of 0.44 against a solver dual
/// infeasibility of 2.3e-8, with their `ipopt_zU_out` matching the gradient
/// to every printed digit.
///
/// This is a *relaxation*: it projects out exactly the component a bound
/// multiplier would carry, so it cannot see a missing or wrong `z` (gh #495).
/// When the `.sol` exports the multipliers, prefer
/// [`exact_dual_infeasibility`].
fn bound_projected_residual(
    s: &[Number],
    x: &[Number],
    x_l: &[Number],
    x_u: &[Number],
    compl_tol: Number,
) -> Number {
    let n = s.len();
    let mut dual_inf = 0.0_f64;
    for j in 0..n {
        let fixed = lower_bound_present(x_l[j])
            && upper_bound_present(x_u[j])
            && (x_u[j] - x_l[j]).abs() <= 1e-12;
        if fixed {
            continue;
        }
        let absorbable = |present: bool, bound: Number, gap: Number, z: Number| {
            present && (gap.abs() <= 1e-8 * (1.0 + bound.abs()) || z * gap.max(0.0) <= compl_tol)
        };
        let r = if s[j] > 0.0 {
            if absorbable(lower_bound_present(x_l[j]), x_l[j], x[j] - x_l[j], s[j]) {
                0.0
            } else {
                s[j]
            }
        } else if absorbable(upper_bound_present(x_u[j]), x_u[j], x_u[j] - x[j], -s[j]) {
            0.0
        } else {
            -s[j]
        };
        dual_inf = dual_inf.max(r);
    }
    dual_inf
}

/// Exact dual infeasibility `‖s − (z_L^suffix + z_U^suffix)‖∞`, with `s` the
/// [`lagrangian_gradient`] at the sign matching the `.sol`'s dual convention.
///
/// Stationarity in pounce's internal convention is
/// `∇f + Jᵀλ − z_L + z_U = 0` with `z_L, z_U ≥ 0`, and the `.sol` suffixes
/// carry `ipopt_zL_out = +z_L`, `ipopt_zU_out = −z_U` — both equal to the
/// objective-gradient component at the bound, matching Ipopt 3.14 (gh #296).
/// So `−z_L + z_U` is exactly `−(zL_out + zU_out)`, and no sign has to be
/// guessed here beyond the one already chosen for `λ`.
///
/// Unlike [`bound_projected_residual`] this sees a bound multiplier that is
/// missing or wrong, because nothing is projected away.
fn exact_dual_infeasibility(s: &[Number], z_l_suf: &[Number], z_u_suf: &[Number]) -> Number {
    let mut dual_inf = 0.0_f64;
    for (j, &s_j) in s.iter().enumerate() {
        let z = z_l_suf.get(j).copied().unwrap_or(0.0) + z_u_suf.get(j).copied().unwrap_or(0.0);
        dual_inf = dual_inf.max((s_j - z).abs());
    }
    dual_inf
}

/// Bound complementarity over **variables**:
/// `max_j max(|z_L·(x−x_L)|, |z_U·(x_U−x)|)` — the quantity Ipopt prints as
/// `Complementarity` (gh #516). Only variables with a finite bound on the
/// side in question contribute.
///
/// Magnitudes throughout, so the result does not depend on which sign
/// convention the writer used for the multipliers, nor on which side of a
/// bound the point sits.
fn bound_complementarity(
    x: &[Number],
    x_l: &[Number],
    x_u: &[Number],
    z_l_suf: &[Number],
    z_u_suf: &[Number],
) -> Number {
    let mut comp = 0.0_f64;
    for j in 0..x.len() {
        if lower_bound_present(x_l[j]) {
            let z = z_l_suf.get(j).copied().unwrap_or(0.0);
            comp = comp.max((z * (x[j] - x_l[j])).abs());
        }
        if upper_bound_present(x_u[j]) {
            let z = z_u_suf.get(j).copied().unwrap_or(0.0);
            comp = comp.max((z * (x_u[j] - x[j])).abs());
        }
    }
    comp
}

/// `max_i |λ_i| · dist(g_i, active side)` over constraints with a finite
/// range — a constraint with a nonzero multiplier should be active.
/// Equalities (`g_l == g_u`) contribute 0. Best-effort, informational.
///
/// This is **constraint** complementarity, over rows, and is not the
/// quantity a solver reports as `Complementarity` — that one is
/// [`bound_complementarity`], over variables. The two are unrelated in
/// magnitude; see the module docs (gh #516).
fn row_complementarity(lambda: &[Number], g: &[Number], g_l: &[Number], g_u: &[Number]) -> Number {
    let mut comp = 0.0_f64;
    for i in 0..lambda.len() {
        // An equality needs *both* bounds present (gh #403): `g_l = g_u = -5e20`
        // is the one-sided `g <= -5e20`, not an equality at `-5e20`, and
        // skipping it here would drop a real complementarity term.
        if lower_bound_present(g_l[i])
            && upper_bound_present(g_u[i])
            && (g_u[i] - g_l[i]).abs() <= 1e-12
        {
            continue; // equality: multiplier is free, no complementarity
        }
        let dl = if lower_bound_present(g_l[i]) {
            (g[i] - g_l[i]).abs()
        } else {
            Number::INFINITY
        };
        let du = if upper_bound_present(g_u[i]) {
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

pub(crate) fn name_at(names: &[String], i: usize, kind: char) -> String {
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

    #[test]
    fn box_violation_rejects_non_finite() {
        // Regression: a fabricated `.sol` carrying NaN must register an
        // infinite violation, not slip through as feasible. Before the
        // `is_finite` guard, `NaN.max(_).max(0.0)` collapsed to `0.0`
        // (f64::max drops NaN operands) and the checker reported VERIFIED.
        assert_eq!(box_violation(Number::NAN, 0.0, 10.0), Number::INFINITY);
        // ±∞ pinned at an unbounded variable is not a real point either.
        assert_eq!(
            box_violation(Number::INFINITY, 0.0, NLP_UPPER_BOUND_INF),
            Number::INFINITY
        );
        assert_eq!(
            box_violation(Number::NEG_INFINITY, NLP_LOWER_BOUND_INF, 10.0),
            Number::INFINITY
        );
    }

    /// **gh #403.** `verify` exists to be the independent check on a `.sol`.
    /// A checker that under-reports is worse than its blast radius suggests.
    ///
    /// `is_finite_bound` was a *band* membership test —
    /// `b > NLP_LOWER_BOUND_INF && b < NLP_UPPER_BOUND_INF` — applied to `lo`
    /// and `hi` alike. A real upper bound of `-5e20` failed it, so `above`
    /// became `-inf` and the violation read `0.0`: **ACCEPTED for a `.sol` that
    /// violates a declared bound.**
    #[test]
    fn a_bound_past_the_opposite_sentinel_still_scores_a_violation() {
        // x <= -5e20, no lower bound. The point 0.0 violates it by 5e20.
        let v = box_violation(0.0, NLP_LOWER_BOUND_INF, -5.0e20);
        assert_eq!(
            v, 5.0e20,
            "0 is 5e20 above an upper bound of -5e20; scoring it 0.0 lets a \
             fabricated .sol past the feasibility gate"
        );
        // Mirror: x >= 5e20, no upper bound.
        assert_eq!(box_violation(0.0, 5.0e20, NLP_UPPER_BOUND_INF), 5.0e20);
        // A point that does satisfy the same bound still scores zero.
        assert_eq!(box_violation(-6.0e20, NLP_LOWER_BOUND_INF, -5.0e20), 0.0);
        // And the sentinels themselves still mean "no bound".
        assert_eq!(
            box_violation(1e30, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF),
            0.0
        );
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

    /// `min (x−3)² + (y+2)²  s.t.  x ≤ 1, y ≥ −1` — the model whose export
    /// convention is pinned in `main.rs` (gh #296): `ipopt_zL_out = +z_L`,
    /// `ipopt_zU_out = −z_U`, both equal to `∂f/∂x` at the bound.
    ///
    /// At the exact optimum every slack is zero, so bound complementarity is
    /// zero whichever sign convention the writer used — the check is on
    /// magnitudes. Off the optimum it is `|z| · slack`.
    #[test]
    fn bound_complementarity_is_z_times_slack_over_variables() {
        let x_l = [NLP_LOWER_BOUND_INF, -1.0];
        let x_u = [1.0, NLP_UPPER_BOUND_INF];
        // Exactly on both bounds: no slack anywhere.
        assert_eq!(
            bound_complementarity(&[1.0, -1.0], &x_l, &x_u, &[0.0, 2.0], &[-4.0, 0.0]),
            0.0
        );
        // Pull x off its upper bound by 1e-3 while keeping z_U: the product
        // is the residual, and the sign of the multiplier does not enter.
        let c = bound_complementarity(&[0.999, -1.0], &x_l, &x_u, &[0.0, 2.0], &[-4.0, 0.0]);
        assert!((c - 4.0e-3).abs() < 1e-12, "got {c}");
        let flipped = bound_complementarity(&[0.999, -1.0], &x_l, &x_u, &[0.0, 2.0], &[4.0, 0.0]);
        assert_eq!(c, flipped, "magnitudes only — no sign convention assumed");
        // A variable with no bound on the side in question contributes
        // nothing, however large its (meaningless) multiplier.
        assert_eq!(
            bound_complementarity(
                &[0.0],
                &[NLP_LOWER_BOUND_INF],
                &[NLP_UPPER_BOUND_INF],
                &[1e6],
                &[1e6]
            ),
            0.0
        );
    }

    /// The exact residual uses the multipliers the `.sol` actually carries,
    /// so it sees what the bound-projected one projects away — the gh #495
    /// blind spot: a bound multiplier that is missing or wrong leaves the
    /// projected residual at `0.0`.
    #[test]
    fn exact_dual_infeasibility_sees_what_the_projection_hides() {
        // `min (x−3)² s.t. x ≤ 1`: x* = 1, ∇f = −4, so z_U = 4 and the
        // exported suffix is `ipopt_zU_out = −4`.
        let s = [-4.0];
        let x = [1.0];
        let x_l = [NLP_LOWER_BOUND_INF];
        let x_u = [1.0];
        assert_eq!(exact_dual_infeasibility(&s, &[0.0], &[-4.0]), 0.0);

        // Projection: x sits on its upper bound, so a valid z_U absorbs the
        // whole negative gradient and the residual reads zero — with *no*
        // multiplier supplied at all.
        assert_eq!(bound_projected_residual(&s, &x, &x_l, &x_u, 1e-6), 0.0);
        // The exact check does not get to assume one exists.
        assert_eq!(exact_dual_infeasibility(&s, &[0.0], &[0.0]), 4.0);
        // Nor that it has the right sign.
        assert_eq!(exact_dual_infeasibility(&s, &[0.0], &[4.0]), 8.0);
    }

    /// An interior-point answer stops `μ/z` short of an active bound, not on
    /// it. Numbers from PGLib case6468_rte: `x` 1.9e-8 inside an upper
    /// bound of 0.2189 with gradient −0.4399 is an active bound carrying
    /// `z_U = 0.44` at complementarity 8.4e-9 — not an interior variable
    /// with a residual of 0.44, which is what a gap-only activity test
    /// (1.2e-8 here) reported. The same gradient on a variable well inside
    /// its box is still a residual.
    #[test]
    fn a_barrier_stand_off_from_a_bound_is_still_active() {
        let x_l = [-0.1276];
        let x_u = [0.2189];
        let s = [-0.4399];
        let near = [0.2189 - 1.9e-8];
        assert_eq!(bound_projected_residual(&s, &near, &x_l, &x_u, 1e-6), 0.0);
        let inside = [0.05];
        assert_eq!(
            bound_projected_residual(&s, &inside, &x_l, &x_u, 1e-6),
            0.4399
        );
        // Only the bound on the matching side absorbs: a positive gradient
        // at the upper bound needs a negative z_U and stays a residual.
        assert_eq!(
            bound_projected_residual(&[0.4399], &near, &x_l, &x_u, 1e-6),
            0.4399
        );
    }

    /// **gh #516.** Constraint complementarity (rows) and bound
    /// complementarity (variables) are different quantities at the same
    /// point, and can disagree by orders of magnitude. Printing either under
    /// a bare `complementarity residual` label invites the comparison that
    /// cost two people an afternoon in #505; this test pins the fact that
    /// makes the label matter.
    #[test]
    fn row_and_bound_complementarity_are_different_quantities() {
        // One inequality row `g ≥ 0`, slack 4.5e-2, multiplier 1 — a real
        // row-complementarity residual.
        let rows = row_complementarity(&[1.0], &[4.5e-2], &[0.0], &[NLP_UPPER_BOUND_INF]);
        assert!((rows - 4.5e-2).abs() < 1e-15);
        // The same point's variables sit hard on their bounds: bound
        // complementarity is eleven orders of magnitude smaller.
        let bounds = bound_complementarity(
            &[1.0],
            &[NLP_LOWER_BOUND_INF],
            &[1.0 + 1e-11],
            &[0.0],
            &[-1.0],
        );
        assert!(bounds < 1e-10, "got {bounds}");
        assert!(
            rows / bounds > 1e8,
            "the two must not be read as one number"
        );
    }

    /// The same predicate sizes a row's magnitude for the scale-relative
    /// feasibility test. A row written at `5e20` must report that magnitude,
    /// not fall back to its evaluated value alone.
    #[test]
    fn row_magnitude_counts_a_bound_past_the_opposite_sentinel() {
        assert_eq!(
            row_magnitude(1.0, NLP_LOWER_BOUND_INF, -5.0e20),
            5.0e20,
            "the row's own upper bound is its magnitude"
        );
        assert_eq!(row_magnitude(1.0, 5.0e20, NLP_UPPER_BOUND_INF), 5.0e20);
        // Absent on both sides: only the evaluated value carries magnitude.
        assert_eq!(
            row_magnitude(3.0, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF),
            3.0
        );
    }
}
