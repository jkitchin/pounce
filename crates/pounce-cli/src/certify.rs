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

use pounce_convex::sos::{PolyProblem, Polynomial, sos_constrained_lower_bound_gram, sos_opts};
use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput};
use pounce_lean_cert::emit_sos::{
    Ball, SosConstraint, SosInput, emit_sos_certificate, sos_problem_block,
};
use pounce_lean_cert::{
    Certificate, canonical_problem, emit_certificate, emit_feasible_certificate,
    emit_infeasible_certificate, emit_unbounded_certificate, problem_block, to_canonical_json,
};
use pounce_nl::nl_reader;

use crate::dispatch::analyze_quadratic_full;
use crate::poly_extract::{PolyObjective, extract_poly_constraints, extract_poly_objective};
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

Supported slices (v1), chosen automatically from the .nl:

  * convex QP -> verdict `global-min`: quadratic objective (PSD Hessian),
    minimize, linear constraints (one-sided, two-sided ranges, or
    equalities), and variable bounds (one-sided, box, or fixed).

  * polynomial -> verdict `global-lower-bound`: an objective of degree > 2.
    POUNCE solves an SOS relaxation and certifies an exact rational bound
    `g <= p(x)` -- with no convexity assumed, which is the point: a KKT
    argument cannot establish global optimality for a nonconvex polynomial.

    With no constraints the bound holds for every x. With polynomial
    inequality constraints (including finite variable bounds) it holds on
    the feasible set, via a Putinar certificate: one sum-of-squares
    multiplier per constraint, plus a bare square, matching `p - g`
    identically. That is what lets a bound exist at all for an objective
    that is unbounded below off the feasible set.

    If an exact rational *feasible* point attains that bound, the verdict
    strengthens to `global-min` and the certificate exhibits the minimizer.
    A minimizer at irrational coordinates leaves the bound unattained; the
    certificate then proves the bound alone, which still holds.

Maximize, non-polynomial objectives, and equality constraints on a
higher-degree problem are refused (exit 2). An equality needs a
sign-unrestricted multiplier rather than an SOS one, and splitting it into
two inequalities leaves the feasible set with empty interior, where Putinar
no longer guarantees a certificate exists at any degree.

With --feasible, the weaker verdict `feasible` is emitted instead: the
reported point violates no constraint by more than the certificate's own
exact tolerance, and a genuinely feasible point exists within that distance
of it. Nothing is claimed about optimality. This is opt-in rather than a
fallback -- a failed optimality certificate is a result worth seeing, not
one to quietly replace with a weaker claim.

With --local the claim is narrowed to a ball around the solution, giving
`local-min` (or `local-lower-bound`). A ball is just one more polynomial
inequality, so this is the same Putinar machinery with one extra multiplier
-- no second-order conditions, no constraint qualification. That matters
because a nonconvex problem usually has local minima that are not global:
there the global claim is FALSE, so no certificate for it exists at any
relaxation order, while the local one is both true and easier to certify.

options:
  -o, --output <path>     write the certificate JSON here (default: stdout)
      --active-tol <eps>  active-set detection tolerance on the float
                          solution (default: 1e-7)
      --feasible          certify feasibility of the reported point instead
                          of optimality
      --local             certify optimality within a ball around the
                          solution instead of globally
      --radius <r>        radius of that ball (default: 1); needs --local
  -h, --help              show this message";

#[derive(Debug)]
struct CertifyArgs {
    nl: PathBuf,
    sol: PathBuf,
    output: Option<PathBuf>,
    active_tol: f64,
    /// Certify feasibility of the reported point instead of optimality.
    feasible: bool,
    /// Radius of the ball a *local* claim is restricted to. `None` is the
    /// unrestricted claim.
    radius: Option<f64>,
}

/// Ball radius `--local` uses when `--radius` is not given.
///
/// There is no principled default — the right radius is problem-specific — so
/// this is chosen to be a round number that produces small exact coefficients
/// and a claim big enough to be interesting. Too large and the SOS certificate
/// stops existing (the ball swallows a better minimum); too small and the claim
/// is true but says little. `--radius` exists because this will often be wrong.
const DEFAULT_RADIUS: f64 = 1.0;

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
    let mut feasible = false;
    let mut local = false;
    let mut radius: Option<f64> = None;
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
            "--feasible" => feasible = true,
            "--local" => local = true,
            "--radius" => {
                let v = it.next().ok_or("--radius requires a value")?;
                radius = Some(v.parse().map_err(|e| format!("--radius: {e}"))?);
            }
            other if other.starts_with('-') => return Err(format!("unknown flag `{other}`")),
            _ => positionals.push(PathBuf::from(arg)),
        }
    }
    // `--radius` without `--local` would silently do nothing, which is the kind
    // of flag that costs someone an afternoon.
    if radius.is_some() && !local {
        return Err("--radius has no meaning without --local".to_string());
    }
    if feasible && local {
        return Err("--feasible and --local ask for different claims; pick one".to_string());
    }
    match positionals.len() {
        2 => Ok(Some(CertifyArgs {
            nl: positionals[0].clone(),
            sol: positionals[1].clone(),
            output,
            active_tol,
            feasible,
            radius: local.then(|| radius.unwrap_or(DEFAULT_RADIUS)),
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

/// Decide whether this `.nl` belongs on the SOS slice rather than the QP one.
///
/// The test is degree, not failure-to-be-a-QP: a degree ≤ 2 objective stays on
/// the QP path even though the SOS machinery could also handle it. That path
/// proves strictly more — it exhibits a minimizer and certifies it *is* the
/// minimum, where SOS only bounds it — so falling back to SOS would silently
/// downgrade the claim for every convex QP whose emitter happened to hiccup.
/// Routing on a property of the problem keeps which theorem you get
/// predictable.
///
/// Returns `None` (rather than an error) whenever the problem is not on the SOS
/// slice, so the caller can fall through to the QP path and report *its*
/// diagnosis — which is the relevant one for, say, a nonlinear constraint.
///
/// The degree test is on the *objective*: a linear objective under polynomial
/// constraints stays on the QP path, which is right, because it is a linear
/// program there and nothing about SOS improves on that.
fn sos_slice(prob: &nl_reader::NlProblem) -> Option<(PolyObjective, PolyFeasibleSet)> {
    let po = extract_poly_objective(prob).ok()?;
    if po.degree() <= 2 {
        return None;
    }
    let g = extract_poly_constraints(prob).ok()?;
    Some((po, g))
}

/// The feasible set as `gₖ(x) ≥ 0` term lists; empty means all of `ℝⁿ`.
type PolyFeasibleSet = Vec<Vec<(Vec<usize>, f64)>>;

fn backend() -> Box<dyn pounce_linsol::SparseSymLinearSolverInterface> {
    Box::new(pounce_feral::FeralSolverInterface::new())
}

/// Run the SOS relaxation and turn its float output into an exact certificate.
///
/// The solver's bound and Gram matrix are hints only — [`emit_sos_certificate`]
/// re-derives an exact `γ` and an exact `G` and refuses if it cannot. So a
/// relaxation that is loose, or a solve that fails outright, costs a
/// certificate; it cannot produce a wrong one.
///
/// `x_float` is the local solve's iterate. It is a *hint*: the emitter tries to
/// snap it to a nearby rational point that attains the exact `γ`, which upgrades
/// the verdict from a bound to a global minimum. A wrong or missing point leaves
/// the bound unattained, never a wrong verdict.
///
/// With `g` non-empty the claim becomes a bound *on the feasible set*, proved by
/// a Putinar identity. Same relaxation, same emitter, same exactness argument —
/// what changes is that the blocks now come with the polynomial each multiplies.
///
/// With `radius` set the claim is narrowed once more, to a ball around a
/// rational point near `x*`. That is what makes a *local* certificate possible
/// at all on a nonconvex problem: at a local minimum that is not global, the
/// unrestricted claim is simply false, so no certificate for it exists at any
/// relaxation order. Restricted to a ball it becomes true — and easier, since a
/// ball is a strong localizer.
fn certify_sos(
    po: &PolyObjective,
    g: &PolyFeasibleSet,
    x_float: &[f64],
    radius: Option<f64>,
    meta: &CertMeta,
) -> Result<Certificate, String> {
    let ball = radius.map(|r| ball_around(po.n, x_float, r)).transpose()?;

    let poly = Polynomial::new(po.n, po.terms.clone());
    let prob = g.iter().fold(PolyProblem::new(poly), |p, gk| {
        p.ge(Polynomial::new(po.n, gk.clone()))
    });
    // The ball goes in last, matching the multiplier order the emitter uses.
    let prob = match ball.as_ref() {
        Some((center, radius_sq)) => prob.ge(Polynomial::new(
            po.n,
            ball_terms_float(po.n, center, *radius_sq),
        )),
        None => prob,
    };
    let (bound, gram) = sos_constrained_lower_bound_gram(&prob, None, &sos_opts(), backend);
    if !bound.lower_bound.is_finite() {
        return Err(format!(
            "the SOS relaxation did not produce a finite bound (status {:?}); \
             nothing to certify",
            bound.status
        ));
    }
    // σ₀ plus one localizing block per inequality, in that order. This is a
    // shape check on our own pipeline rather than on user input — but the
    // certificate's `multiplier` indices are built from this alignment, so it is
    // worth failing loudly rather than emitting a mispaired identity for the
    // exact rounder to reject with a much vaguer message.
    let Some((sigma0, localizing)) = gram.split_first() else {
        return Err("the SOS relaxation returned no Gram blocks".to_string());
    };
    let expected = g.len() + usize::from(ball.is_some());
    if localizing.len() != expected {
        return Err(format!(
            "expected {expected} localizing block(s) for {} constraint(s){}, got {}",
            g.len(),
            if ball.is_some() {
                " plus a neighborhood"
            } else {
                ""
            },
            localizing.len()
        ));
    }
    // The ball's block is the last one, because that is where it was appended.
    let (con_blocks, ball_block) = localizing.split_at(g.len());
    emit_sos_certificate(
        &SosInput {
            n: po.n,
            terms: po.terms.clone(),
            basis: sigma0.basis.clone(),
            gram_float: sigma0.matrix.clone(),
            constraints: g
                .iter()
                .zip(con_blocks)
                .map(|(gk, blk)| SosConstraint {
                    g: gk.clone(),
                    basis: blk.basis.clone(),
                    gram_float: blk.matrix.clone(),
                })
                .collect(),
            bound_float: bound.lower_bound,
            x_float: x_float.to_vec(),
            neighborhood: ball.map(|(center, radius_sq)| Ball {
                center,
                radius_sq,
                basis: ball_block[0].basis.clone(),
                gram_float: ball_block[0].matrix.clone(),
            }),
        },
        meta,
    )
    .map_err(|e| format!("cannot certify a bound for this polynomial: {e}"))
}

/// Grids the ball's center is snapped to, coarsest first.
const CENTER_DENOMS: [i64; 8] = [1, 2, 3, 4, 8, 64, 1024, 1 << 20];

/// Choose the ball for a local claim: a rational center near `x*`, and `r²`.
///
/// The center is *snapped*, and deliberately to the coarsest grid that still
/// holds `x*` well inside the ball (within half the radius). It does not need to
/// be the minimizer — it only needs to contain it — and a coarse center keeps
/// the ball's expanded coefficients small, which keeps both the SDP well
/// conditioned and the generated Lean readable. Snapping to the raw f64 would
/// give a center with a 2⁵²-ish denominator and a correspondingly ugly theorem.
///
/// Nothing here can make a certificate wrong: whatever ball comes out, that ball
/// is what the emitted theorem quantifies over.
fn ball_around(n: usize, x_float: &[f64], radius: f64) -> Result<(Vec<f64>, f64), String> {
    if x_float.len() != n {
        return Err(format!(
            "--local needs a solution point of length {n}, got {}",
            x_float.len()
        ));
    }
    if !radius.is_finite() || radius <= 0.0 {
        return Err("--radius must be a finite positive number".to_string());
    }
    if let Some(bad) = x_float.iter().find(|v| !v.is_finite()) {
        return Err(format!(
            "--local needs a finite solution point; found {bad} in the .sol"
        ));
    }
    let radius_sq = radius * radius;
    for d in CENTER_DENOMS {
        let center: Vec<f64> = x_float
            .iter()
            .map(|v| (v * d as f64).round() / d as f64)
            .collect();
        let dist_sq: f64 = center
            .iter()
            .zip(x_float)
            .map(|(c, v)| (c - v) * (c - v))
            .sum();
        if center.iter().all(|c| c.is_finite()) && dist_sq * 4.0 <= radius_sq {
            return Ok((center, radius_sq));
        }
    }
    // Every grid was too coarse for this radius, so use `x*` itself: it is an
    // f64 and therefore already rational, just an inconvenient one.
    Ok((x_float.to_vec(), radius_sq))
}

/// `r² − Σⱼ(xⱼ − cⱼ)²` in f64, for the SDP's benefit only.
///
/// The exact expansion the *claim* rests on is `pounce_lean_cert::ball_terms`,
/// computed over ℚ from the same center and radius. This one feeds the float
/// relaxation, which only ever produces a Gram hint, so rounding here costs at
/// most a failed search.
fn ball_terms_float(n: usize, center: &[f64], radius_sq: f64) -> Vec<(Vec<usize>, f64)> {
    let mut terms: Vec<(Vec<usize>, f64)> = Vec::with_capacity(2 * n + 1);
    let mut constant = radius_sq;
    for (j, cj) in center.iter().enumerate() {
        constant -= cj * cj;
        let mut lin = vec![0usize; n];
        lin[j] = 1;
        terms.push((lin, 2.0 * cj));
        let mut quad = vec![0usize; n];
        quad[j] = 2;
        terms.push((quad, -1.0));
    }
    terms.push((vec![0usize; n], constant));
    terms
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

    // AMPL reserves solve_result_num 200-299 for "infeasible". That verdict is
    // not a claim about `parsed.x` at all — it is certified by a Farkas ray, so
    // it routes to a different emitter.
    let infeasible = matches!(parsed.solve_result_num, Some(k) if (200..300).contains(&k));
    // 300-399 is AMPL's "unbounded". The diverging primal iterate is both the
    // feasible witness and the recession direction; see emit_unbounded_certificate.
    let unbounded = matches!(parsed.solve_result_num, Some(k) if (300..400).contains(&k));

    let dual_ray = parsed.lambda.clone();
    let iterate = parsed.x.clone();
    let meta = CertMeta {
        nl_sha256,
        sol_sha256,
        solver: format!("pounce {}", env!("CARGO_PKG_VERSION")),
    };

    // A higher-degree polynomial gets a *bound*, not a claim about `parsed.x`.
    // Checked before the QP extraction, which would reject it for being degree
    // > 2 — and reject it with the wrong diagnosis, since such a problem is
    // certifiable, just by a different theorem.
    //
    // The infeasible/unbounded verdicts take precedence: they are statements
    // about the solve, and neither has an SOS reading (an SOS bound says
    // nothing about whether the feasible set is empty, and an unbounded
    // problem has no lower bound to certify).
    if !infeasible
        && !unbounded
        && let Some((po, g)) = sos_slice(&prob)
    {
        if args.feasible {
            return Err(if g.is_empty() {
                // Every point is feasible for an unconstrained polynomial, so
                // the certificate would prove nothing. Say so rather than emit
                // it.
                "--feasible: this problem is an unconstrained polynomial, so feasibility \
                 is vacuous; drop the flag to certify the objective bound instead"
                    .to_string()
            } else {
                // The `feasible` verdict is proved by exact projection onto the
                // active constraints, which needs them linear. Nothing here is
                // wrong with the problem — the emitter just does not cover it.
                "--feasible: the feasible verdict covers linear constraints only, and this \
                 problem is on the polynomial slice; drop the flag to certify a bound on \
                 the feasible set instead"
                    .to_string()
            });
        }
        let cert = certify_sos(&po, &g, &iterate, args.radius, &meta)?;
        return to_canonical_json(&cert).map_err(|e| format!("serialization failed: {e}"));
    }

    // `--local` is only meaningful where a claim can be *restricted*, and the
    // polynomial slice is the only one that admits one. On the QP path a KKT
    // certificate already proves global optimality, so narrowing it to a ball
    // would be strictly weaker for no gain.
    if args.radius.is_some() {
        return Err(
            "--local applies to the polynomial slice (objective degree > 2); this problem \
             is a convex QP, where the certificate already proves global optimality"
                .to_string(),
        );
    }

    let input = nl_to_qp_input(&prob, parsed.x, args.active_tol)?;

    if args.feasible {
        if infeasible {
            // The two claims are contradictory. `emit_feasible_certificate`
            // would refuse anyway, but on a projection failure deep inside the
            // exact arithmetic — a needlessly obscure way to report that the
            // solver already answered this question.
            return Err(
                "--feasible: the solve terminated infeasible, so there is no feasible \
                 point to certify; drop the flag for a Farkas certificate of that"
                    .to_string(),
            );
        }
        let cert = emit_feasible_certificate(&input, &meta)
            .map_err(|e| format!("cannot certify this point feasible: {e}"))?;
        return to_canonical_json(&cert).map_err(|e| format!("serialization failed: {e}"));
    }

    let cert = if infeasible {
        if dual_ray.len() != input.constraints.len() {
            return Err(format!(
                "infeasible solve carries {} duals but the problem has {} constraints                  (is this the right .sol for this .nl?)",
                dual_ray.len(),
                input.constraints.len()
            ));
        }
        // The ray's sign convention does not matter here. `refine_farkas` reads
        // only the support (via magnitudes) and then orients the exact ray
        // itself so that `b·y > 0` — so the AMPL-vs-pounce dual sign ambiguity
        // that `pounce verify` has to resolve by trying both simply does not
        // arise on this path.
        emit_infeasible_certificate(&input, &meta, &dual_ray, args.active_tol)
            .map_err(|e| format!("cannot certify this infeasible solve: {e}"))?
    } else if unbounded {
        emit_unbounded_certificate(&input, &meta, &iterate)
            .map_err(|e| format!("cannot certify this unbounded solve: {e}"))?
    } else {
        emit_certificate(&input, &meta).map_err(|e| format!("cannot certify this solve: {e}"))?
    };
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

    // Which half of the `problem` block to re-derive follows from the cert's own
    // `problem_class`. Trusting that field is safe *because* of what follows: a
    // cert claiming `sos-poly` for a QP re-derives a polynomial that will not
    // match, so the class is checked by the comparison rather than assumed.
    let p_nl = match cert.problem_class.as_str() {
        "sos-poly" => {
            let po = extract_poly_objective(&prob)
                .map_err(|e| format!("cannot re-derive the polynomial from this .nl: {e}"))?;
            let g = extract_poly_constraints(&prob)
                .map_err(|e| format!("cannot re-derive the feasible set from this .nl: {e}"))?;
            // The neighborhood of a local claim is the *certificate's* choice,
            // not something the `.nl` determines — no re-derivation could
            // reproduce it, and it is not supposed to. So it is carried across
            // rather than re-derived, and the comparison below then checks
            // exactly what the `.nl` does determine: the polynomial and the
            // feasible set. `sos_problem_block` still validates it (length `n`,
            // radius² > 0), so a malformed one is refused here rather than
            // reaching the codegen.
            // It is carried exactly, not through f64: a rational center with an
            // awkward denominator would not survive the round trip, and the
            // comparison below is byte-for-byte.
            sos_problem_block(po.n, &po.terms, &g, cert.problem.neighborhood.as_ref())
                .map_err(|e| format!("cannot re-derive problem: {e}"))?
        }
        _ => {
            let input = nl_to_qp_input(&prob, vec![0.0; n], 0.0)?; // x_float unused
            problem_block(&input).map_err(|e| format!("cannot re-derive problem: {e}"))?
        }
    };
    if canonical_problem(&p_nl) != canonical_problem(&cert.problem) {
        return Err("certificate describes a different problem than this .nl \
             (objective/constraints/bounds mismatch)"
            .to_string());
    }
    Ok(())
}
