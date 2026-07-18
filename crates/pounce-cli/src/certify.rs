//! `pounce certify <problem.nl> <claim.sol>` — emit an exact-rational
//! `pounce.lean-cert/v1` certificate for a convex-QP / `global-min` solve.
//!
//! This is the I/O + classification glue around [`pounce_lean_cert`]: read and
//! hash the `.nl`/`.sol` (content-addressing, exactly as `pounce verify`),
//! extract the quadratic objective and linear constraints from the `.nl`, hand
//! the neutral `f64` problem to the emitter, and write the certificate. The
//! emitter does the exact-rational work and refuses anything off the supported
//! slice — so this layer only translates POUNCE's data, it never decides
//! soundness.

use std::path::PathBuf;
use std::process::ExitCode;

use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput};
use pounce_lean_cert::{
    Certificate, canonical_problem, emit_certificate, problem_block, to_canonical_json,
};
use pounce_nl::nl_reader;

use crate::dispatch::analyze_quadratic_full;
use crate::verify::{parse_sol, sha256};

/// `nl_reader` encodes "no bound" as the AMPL sentinel `±1e19` (see
/// `parse_bound_line`), not `f64::INFINITY`. The certificate's neutral input
/// uses true infinities, so collapse the sentinel here at the `.nl` boundary.
const AMPL_INF: f64 = 1e19;
fn deinf(x: f64) -> f64 {
    if x >= AMPL_INF {
        f64::INFINITY
    } else if x <= -AMPL_INF {
        f64::NEG_INFINITY
    } else {
        x
    }
}

const USAGE: &str = "\
usage: pounce certify <problem.nl> <claim.sol> [options]

Emit an exact-rational pounce.lean-cert/v1 certificate that the pounce-lean
repo can turn into a kernel-checked Lean proof of global optimality.

Supported slice (v1): convex QP (quadratic objective, PSD Hessian),
minimize, linear constraints (one-sided, two-sided ranges, or equalities),
and variable bounds (one-sided, box, or fixed). Nonconvex / maximize /
nonlinear inputs are refused (exit 2).

options:
  -o, --output <path>     write the certificate JSON here (default: stdout)
      --active-tol <eps>  active-set detection tolerance on the float
                          solution (default: 1e-7)
  -h, --help              show this message";

#[derive(Debug)]
struct CertifyArgs {
    nl: PathBuf,
    sol: PathBuf,
    output: Option<PathBuf>,
    active_tol: f64,
}

pub fn run_from_argv(rest: &[String]) -> ExitCode {
    let args = match parse_argv(rest) {
        Ok(Some(a)) => a,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("pounce certify: {msg}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(json) => {
            match &args.output {
                Some(path) => {
                    if let Err(e) = std::fs::write(path, json.as_bytes()) {
                        eprintln!("pounce certify: cannot write {}: {e}", path.display());
                        return ExitCode::from(2);
                    }
                }
                None => println!("{json}"),
            }
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("pounce certify: {msg}");
            ExitCode::from(2)
        }
    }
}

fn parse_argv(rest: &[String]) -> Result<Option<CertifyArgs>, String> {
    let mut output = None;
    let mut active_tol = 1e-7;
    let mut positionals: Vec<PathBuf> = Vec::new();
    let mut it = rest.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "-o" | "--output" => {
                let v = it.next().ok_or("--output requires a value")?;
                output = Some(PathBuf::from(v));
            }
            "--active-tol" => {
                let v = it.next().ok_or("--active-tol requires a value")?;
                active_tol = v.parse().map_err(|e| format!("--active-tol: {e}"))?;
            }
            other if other.starts_with('-') => return Err(format!("unknown flag `{other}`")),
            _ => positionals.push(PathBuf::from(arg)),
        }
    }
    match positionals.len() {
        2 => Ok(Some(CertifyArgs {
            nl: positionals[0].clone(),
            sol: positionals[1].clone(),
            output,
            active_tol,
        })),
        _ => Err("expected two positional arguments: <problem.nl> <claim.sol>".to_string()),
    }
}

/// Extract the QP problem fields from a parsed `.nl` (the **Frontend**'s `.nl`
/// half), pairing them with a primal `x_float` hint. Shared by `certify` (real
/// `x*` from the `.sol`) and `cert-verify` (a dummy `x*`, since the problem
/// block ignores it). Errors out on anything off the supported slice.
fn nl_to_qp_input(
    prob: &pounce_nl::nl_reader::NlProblem,
    x_float: Vec<f64>,
    active_tol: f64,
) -> Result<QpInput, String> {
    let n = prob.n;
    let m = prob.m;
    if !prob.minimize {
        return Err("certify supports minimize objectives only (v1)".to_string());
    }

    // --- objective: read it as a quadratic form (Q, c, constant) ---
    let (hess, obj_lin_folded, obj_const_folded) = analyze_quadratic_full(&prob.obj_nonlinear, n)
        .ok_or(
        "objective is not a polynomial of degree ≤ 2 (certify supports convex QP only)",
    )?;
    // The Hessian map is upper-triangular (i ≤ j); the cert stores Q's lower
    // triangle (i ≥ j). These second-partials are exactly the cert Q with
    // half_quadratic = true (f = ½·xᵀQx + …), matching POUNCE's convention.
    let mut q_lower: Vec<(usize, usize, f64)> =
        hess.iter().map(|(&(a, b), &v)| (b, a, v)).collect();
    q_lower.sort_by_key(|&(i, j, _)| (i, j));

    let mut c = vec![0.0f64; n];
    for &(i, v) in &prob.obj_linear {
        c[i] += v;
    }
    for &(i, v) in &obj_lin_folded {
        c[i] += v;
    }
    let constant = prob.obj_constant + obj_const_folded;

    // --- constraints: each must be linear; keep the original range form ---
    let mut constraints = Vec::with_capacity(m);
    for i in 0..m {
        let (chess, clin, cconst) = analyze_quadratic_full(&prob.con_nonlinear[i], n)
            .ok_or_else(|| format!("constraint {i} is not a polynomial of degree ≤ 2"))?;
        if !chess.is_empty() {
            return Err(format!(
                "constraint {i} is nonlinear (quadratic or higher); off the supported QP slice"
            ));
        }
        let mut coeffs = vec![0.0f64; n];
        for &(j, v) in &prob.con_linear[i] {
            coeffs[j] += v;
        }
        for &(j, v) in &clin {
            coeffs[j] += v;
        }
        // A folded constant shifts the bounds: g_l ≤ a·x + k ≤ g_u  ⇔
        // g_l − k ≤ a·x ≤ g_u − k (an inf bound stays inf).
        let lower = deinf(prob.g_l[i]) - cconst;
        let upper = deinf(prob.g_u[i]) - cconst;
        let name = prob
            .con_names
            .get(i)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("c{i}"));
        constraints.push(LinearConstraint {
            name,
            coeffs,
            lower,
            upper,
        });
    }

    Ok(QpInput {
        n,
        q_lower,
        half_quadratic: true,
        c,
        constant,
        constraints,
        var_lower: prob.x_l.iter().copied().map(deinf).collect(),
        var_upper: prob.x_u.iter().copied().map(deinf).collect(),
        x_float,
        active_tol,
    })
}

fn run(args: &CertifyArgs) -> Result<String, String> {
    // --- read + content-address the two inputs ---
    let nl_bytes =
        std::fs::read(&args.nl).map_err(|e| format!("cannot read {}: {e}", args.nl.display()))?;
    let sol_bytes =
        std::fs::read(&args.sol).map_err(|e| format!("cannot read {}: {e}", args.sol.display()))?;
    let nl_sha256 = sha256::hex(&nl_bytes);
    let sol_sha256 = sha256::hex(&sol_bytes);

    let prob = nl_reader::read_nl_file(&args.nl)?;
    let n = prob.n;

    // --- claimed solution: only the primal is needed (duals are recomputed) ---
    let parsed = parse_sol(&String::from_utf8_lossy(&sol_bytes))?;
    if parsed.x.len() != n {
        return Err(format!(
            "solution has {} primal values but the problem has {n} variables \
             (is this the right .sol for this .nl?)",
            parsed.x.len()
        ));
    }

    let input = nl_to_qp_input(&prob, parsed.x, args.active_tol)?;
    let meta = CertMeta {
        nl_sha256,
        sol_sha256,
        solver: format!("pounce {}", env!("CARGO_PKG_VERSION")),
    };

    let cert =
        emit_certificate(&input, &meta).map_err(|e| format!("cannot certify this solve: {e}"))?;
    to_canonical_json(&cert).map_err(|e| format!("serialization failed: {e}"))
}

const VERIFY_USAGE: &str = "\
usage: pounce cert-verify <problem.nl> <cert.json>

Check that a pounce.lean-cert/v1 certificate concerns THIS .nl, by re-deriving
the problem from the .nl and comparing it to the certificate's `problem` block.
This is the consumer-side binding check: it makes `lake build` + the hash
binding sufficient by ruling out a certificate that proves an easier problem
under the real .nl's hash. (It does NOT run Lean — that is the pounce-lean half.)

Exit 0 if the certificate matches this .nl; exit 2 otherwise.

options:
  -h, --help   show this message";

pub fn run_verify_from_argv(rest: &[String]) -> ExitCode {
    let mut positionals: Vec<PathBuf> = Vec::new();
    for arg in rest {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{VERIFY_USAGE}");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("pounce cert-verify: unknown flag `{other}`");
                eprintln!("{VERIFY_USAGE}");
                return ExitCode::from(2);
            }
            _ => positionals.push(PathBuf::from(arg)),
        }
    }
    if positionals.len() != 2 {
        eprintln!("pounce cert-verify: expected <problem.nl> <cert.json>");
        eprintln!("{VERIFY_USAGE}");
        return ExitCode::from(2);
    }
    match verify(&positionals[0], &positionals[1]) {
        Ok(()) => {
            println!("cert-verify: OK — certificate matches this .nl");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("cert-verify: REJECT — {msg}");
            ExitCode::from(2)
        }
    }
}

fn verify(nl: &PathBuf, cert_path: &PathBuf) -> Result<(), String> {
    let nl_bytes = std::fs::read(nl).map_err(|e| format!("cannot read {}: {e}", nl.display()))?;
    let cert_bytes = std::fs::read(cert_path)
        .map_err(|e| format!("cannot read {}: {e}", cert_path.display()))?;
    let cert: Certificate =
        serde_json::from_slice(&cert_bytes).map_err(|e| format!("malformed certificate: {e}"))?;

    // (1) Provenance pre-check: the cert names THIS .nl's bytes.
    let nl_sha256 = sha256::hex(&nl_bytes);
    if cert.binding.nl_sha256 != nl_sha256 {
        return Err(format!(
            "binding.nl_sha256 does not match this .nl\n         cert: {}\n         .nl : {}",
            cert.binding.nl_sha256, nl_sha256
        ));
    }

    // (2) Load-bearing check: re-derive the problem from THIS .nl (the trusted,
    //     deterministic Frontend) and compare to the cert's problem block. A
    //     certificate that proves an easier problem under the real hash fails here.
    let prob = nl_reader::read_nl_file(nl)?;
    let n = prob.n;
    let input = nl_to_qp_input(&prob, vec![0.0; n], 0.0)?; // x_float unused by problem_block
    let p_nl = problem_block(&input).map_err(|e| format!("cannot re-derive problem: {e}"))?;
    if canonical_problem(&p_nl) != canonical_problem(&cert.problem) {
        return Err("certificate describes a different problem than this .nl \
             (objective/constraints/bounds mismatch)"
            .to_string());
    }
    Ok(())
}
