//! End-to-end: a convex-QP `.nl` file routed through the CLI dispatch to
//! the `pounce-convex` interior-point solver (Phase 2 wiring).
//!
//! Fixture `convex_qp.nl` is `min x0² + x1²  s.t.  x0 + x1 = 2`, whose
//! optimum is (1, 1) with objective 2. The tests check that:
//!   - `solver_selection=auto` classifies it as a convex QP and routes
//!     it to the convex IPM (banner names pounce-convex),
//!   - `solver_selection=qp-ipm` (forced) also solves it,
//!   - the `.sol` primal matches the known optimum,
//!   - `solver_selection=nlp` still solves the same file (no regression /
//!     same answer via the general path).

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    fixture_named("convex_qp.nl")
}

fn fixture_named(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// A primal-infeasible convex QP (`x0+x1=1` and `x0+x1=2`) routed to the
/// convex IPM must report infeasible — the HSDE-style verified
/// detection, surfaced end-to-end — and exit non-zero.
#[test]
fn infeasible_qp_reports_infeasible() {
    let out = Command::new(pounce_exe())
        .arg(fixture_named("infeasible_qp.nl"))
        .arg("--no-sol")
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.to_lowercase().contains("infeasible"),
        "expected infeasible status; stdout=\n{stdout}"
    );
    assert_ne!(out.status.code(), Some(0), "infeasible must exit non-zero");
}

/// H4 regression: the AMPL solver-protocol exit-code contract on the convex
/// path. Under `-AMPL` the termination is conveyed through the `.sol`'s
/// `solve_result_num`, so the process must exit 0 even for a non-fatal
/// *unsuccessful* outcome (infeasible/unbounded/limit) — a non-zero exit makes
/// Pyomo / the ASL interface raise `ApplicationError` and never parse the
/// `.sol`. The NLP path already honors this; the convex QP/SOCP paths
/// previously ignored `args.ampl` and returned exit 1 on infeasible, breaking
/// the Pyomo integration on every default-routed LP/convex-QP `.nl`. This test
/// runs both arms on the same infeasible QP: `-AMPL` exits 0 (with the verdict
/// `.sol` written), plain CLI exits non-zero.
#[test]
fn ampl_mode_honors_exit_code_contract_on_infeasible_convex_qp() {
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_h4_ampl_infeasible.sol");
    let _ = std::fs::remove_file(&sol);

    // `-AMPL` arm: exit 0, verdict in the `.sol`.
    let ampl = Command::new(pounce_exe())
        .arg(fixture_named("infeasible_qp.nl"))
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(
        ampl.status.code(),
        Some(0),
        "-AMPL infeasible must exit 0 (verdict travels in the .sol); stdout=\n{}",
        String::from_utf8_lossy(&ampl.stdout)
    );
    let text = std::fs::read_to_string(&sol).expect("verdict .sol written under -AMPL");
    assert!(
        text.contains("200"),
        "the infeasible solve_result_num (200) must be in the .sol:\n{text}"
    );
    let _ = std::fs::remove_file(&sol);

    // Plain-CLI arm on the same problem: still exits non-zero (the contract
    // change is scoped to `-AMPL`).
    let plain = Command::new(pounce_exe())
        .arg(fixture_named("infeasible_qp.nl"))
        .arg("--no-sol")
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_ne!(
        plain.status.code(),
        Some(0),
        "plain-CLI infeasible must still exit non-zero"
    );
}

// --- A2: a forced solver_selection that does not match the detected
// class must error end-to-end (nonzero exit, clear message) and NEVER
// silently mis-solve to a wrong "optimal". `auto` on the same file must
// route safely instead. ---

/// The highest-risk mis-route: forcing the convex QP IPM onto a genuinely
/// *nonconvex* QP (`min x0·x1`, indefinite Hessian). It must error, naming
/// the detected class and the forced solver, and must NOT print an
/// "Optimal Solution Found" — a confident wrong answer is the failure mode
/// this whole effort exists to prevent.
#[test]
fn forced_qp_ipm_on_nonconvex_qp_errors() {
    let out = Command::new(pounce_exe())
        .arg(fixture_named("nonconvex_qp.nl"))
        .arg("--no-sol")
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(2), "forced mismatch must exit 2");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("nonconvex QP") && combined.contains("qp-ipm"),
        "error must name detected class and forced solver:\n{combined}"
    );
    assert!(
        !combined.contains("Optimal Solution Found"),
        "a mismatch must never report a solve:\n{combined}"
    );
}

/// Same nonconvex QP forced to the active-set QP solver: also a mismatch,
/// also must error rather than mis-solve.
#[test]
fn forced_qp_active_set_on_nonconvex_qp_errors() {
    let out = Command::new(pounce_exe())
        .arg(fixture_named("nonconvex_qp.nl"))
        .arg("--no-sol")
        .arg("solver_selection=qp-active-set")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(2));
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("nonconvex QP") && combined.contains("qp-active-set"),
        "error must name detected class and forced solver:\n{combined}"
    );
    assert!(!combined.contains("Optimal Solution Found"), "{combined}");
}

/// Forcing the LP IPM onto a convex *QP* (not an LP): the QP IPM accepts a
/// QP but the LP entry point does not, so this must error too.
#[test]
fn forced_lp_ipm_on_convex_qp_errors() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=lp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(2));
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("convex QP") && combined.contains("lp-ipm"),
        "error must name detected class and forced solver:\n{combined}"
    );
    assert!(!combined.contains("Optimal Solution Found"), "{combined}");
}

/// The safe counterpart: `auto` on the same nonconvex QP must NOT route to
/// the convex IPM. It falls back to the general NLP path and solves to a
/// local optimum (exit 0), so the user gets a sound answer rather than an
/// error or a wrong "global" one.
#[test]
fn auto_routes_nonconvex_qp_to_nlp_safely() {
    let out = Command::new(pounce_exe())
        .arg(fixture_named("nonconvex_qp.nl"))
        .arg("--no-sol")
        .arg("solver_selection=auto")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "auto should solve via NLP");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pounce-nlp") && !stdout.contains("pounce-convex"),
        "auto must fall back to the NLP path, not the convex IPM:\n{stdout}"
    );
    assert!(
        stdout.contains("Optimal Solution Found"),
        "NLP fallback should solve to a local optimum:\n{stdout}"
    );
}

#[test]
fn auto_routes_convex_qp_to_pounce_convex() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=auto")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "should solve");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pounce-convex"),
        "auto should route the convex QP to pounce-convex; stdout=\n{stdout}"
    );
    assert!(
        stdout.contains("Optimal Solution Found"),
        "should report optimal; stdout=\n{stdout}"
    );
}

/// pounce#186: `max_iter=0` on the convex-routed path must honor AMPL/Ipopt
/// semantics — zero iterations cannot reach optimality. The convex path used
/// to report "Optimal Solution Found" regardless (the routed QP was solved by
/// presolve / a direct step that ignored the cap), so a Pyomo `max_iter=0`
/// silently "succeeded". Both arms are checked on the same `auto`-routed convex
/// QP: the plain-CLI status line reports the iteration limit (not optimal), and
/// under `-AMPL` the `.sol` carries the non-optimal `solve_result_num` 400.
#[test]
fn max_iter_zero_on_convex_path_is_not_optimal() {
    // Plain-CLI arm: the convex banner must report the iteration limit.
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=auto")
        .arg("max_iter=0")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pounce-convex"),
        "the QP must still route to the convex path:\n{stdout}"
    );
    assert!(
        !stdout.contains("Optimal Solution Found"),
        "max_iter=0 must not report optimal on the convex path:\n{stdout}"
    );
    assert!(
        stdout.contains("Maximum iterations exceeded"),
        "max_iter=0 must report the iteration limit:\n{stdout}"
    );

    // `-AMPL` arm: the non-optimal verdict travels in the `.sol` as
    // solve_result_num 400, which is what Pyomo reads back.
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_186_max_iter_zero.sol");
    let _ = std::fs::remove_file(&sol);
    let ampl = Command::new(pounce_exe())
        .arg(fixture())
        .arg("-AMPL")
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=auto")
        .arg("max_iter=0")
        .output()
        .expect("spawn pounce");
    assert_eq!(
        ampl.status.code(),
        Some(0),
        "-AMPL must exit 0 (verdict travels in the .sol); stdout=\n{}",
        String::from_utf8_lossy(&ampl.stdout)
    );
    let text = std::fs::read_to_string(&sol).expect("verdict .sol written under -AMPL");
    assert!(
        text.contains("400"),
        "the iteration-limit solve_result_num (400) must be in the .sol:\n{text}"
    );
    let _ = std::fs::remove_file(&sol);
}

#[test]
fn forced_qp_ipm_solves() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("pounce-convex"), "stdout=\n{stdout}");
}

/// The `qp-active-set` route is wired: it dispatches the convex QP to the
/// active-set SQP engine (pounce-qp QP subproblems), not the IPM. The banner
/// must name the active-set solver and the solve must succeed. (Previously the
/// flag was validated then silently fell through to the NLP IPM.)
#[test]
fn forced_qp_active_set_solves_convex_qp() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=qp-active-set")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "active-set route should solve");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("active-set QP (pounce-qp)"),
        "banner must name the active-set solver, not fall through:\n{stdout}"
    );
    assert!(
        stdout.contains("Optimal Solution Found"),
        "active-set route should report optimal:\n{stdout}"
    );
}

/// The active-set route's `.sol` must carry the *real* primal and dual — not
/// the zero fallback. Its solve bypasses the IPM-only `on_converged` capture,
/// so the CLI backfills the solution from `finalize_solution`; this test pins
/// that the captured `x ≈ (1,1)` and the equality dual `≈ −2` match the IPM /
/// NLP convention on the same `min x0²+x1² s.t. x0+x1=2` fixture.
#[test]
fn qp_active_set_sol_matches_known_optimum_and_dual() {
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_qp_active_set_test.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=qp-active-set")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let text = std::fs::read_to_string(&sol).expect("read .sol");
    let floats: Vec<f64> = text
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    // Two primal values ≈ 1.0 (the real solution, not the zero fallback).
    let near_one = floats.iter().filter(|v| (**v - 1.0).abs() < 1e-5).count();
    assert!(
        near_one >= 2,
        "active-set .sol must carry the real primal x ≈ (1,1), not zeros:\n{text}"
    );
    // The equality multiplier is −2 in the same convention as the IPM/NLP path.
    let dual_near = floats
        .iter()
        .copied()
        .min_by(|a, b| (a + 2.0).abs().partial_cmp(&(b + 2.0).abs()).unwrap())
        .expect("a float in .sol");
    assert!(
        (dual_near + 2.0).abs() < 1e-5,
        "active-set equality dual {dual_near} != −2:\n{text}"
    );
}

#[test]
fn nlp_path_still_solves_same_file() {
    // No regression: the general NLP path must still handle the file.
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=nlp")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Optimal Solution Found"),
        "NLP path stdout=\n{stdout}"
    );
}

#[test]
fn sol_primal_matches_known_optimum() {
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_convex_qp_test.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=auto")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let text = std::fs::read_to_string(&sol).expect("read .sol");
    // The primal block lists x0 then x1, each ≈ 1.0. Parse the trailing
    // floats and check the two that are closest to 1.0 are present.
    let near_one = text
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .filter(|v| (v - 1.0).abs() < 1e-5)
        .count();
    assert!(
        near_one >= 2,
        "expected two primal values ≈ 1.0 in .sol:\n{text}"
    );
}

/// The convex QP path's recovered constraint dual must match the NLP
/// path's dual on the same `.nl` file (the reference convention). For
/// `min x0²+x1² s.t. x0+x1=2` the equality multiplier is −2.
#[test]
fn qp_and_nlp_duals_agree() {
    let dir = std::env::temp_dir();

    let run = |sel: &str, out: &std::path::Path| {
        let _ = std::fs::remove_file(out);
        let status = Command::new(pounce_exe())
            .arg(fixture())
            .arg("--sol-output")
            .arg(out)
            .arg(format!("solver_selection={sel}"))
            .output()
            .expect("spawn pounce");
        assert_eq!(status.status.code(), Some(0), "{sel} failed");
        std::fs::read_to_string(out).expect("read .sol")
    };

    // The single constraint dual is the value closest to −2 in each
    // `.sol`'s float block.
    let dual_near = |text: &str| -> f64 {
        text.lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .min_by(|a, b| (a - (-2.0)).abs().partial_cmp(&(b - (-2.0)).abs()).unwrap())
            .expect("a float in .sol")
    };

    let qp_sol = run("qp-ipm", &dir.join("pounce_dual_qp.sol"));
    let nlp_sol = run("nlp", &dir.join("pounce_dual_nlp.sol"));

    let qp_dual = dual_near(&qp_sol);
    let nlp_dual = dual_near(&nlp_sol);
    assert!((qp_dual - (-2.0)).abs() < 1e-5, "QP dual {qp_dual} != −2");
    assert!(
        (qp_dual - nlp_dual).abs() < 1e-5,
        "QP dual {qp_dual} disagrees with NLP dual {nlp_dual}"
    );
}

/// The convex-QP path emits a `pounce.solve-report/v1` JSON report
/// (`--json-output`), matching the schema the NLP path produces — so the
/// benchmark harness can compare QP and NLP solves uniformly. Validates the
/// schema, status, objective, problem dimensions, and iteration count.
#[test]
fn qp_path_emits_json_report() {
    let dir = std::env::temp_dir();
    let json = dir.join("pounce_convex_qp_report.json");
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "QP solve should succeed");

    let text = std::fs::read_to_string(&json).expect("JSON report should be written");
    let report: SolveReport = serde_json::from_str(&text).expect("deserialize report");

    assert_eq!(report.schema, "pounce.solve-report/v1");
    // min x0²+x1² s.t. x0+x1=2 → optimum (1,1), objective 2.
    assert!(
        (report.solution.objective - 2.0).abs() < 1e-5,
        "objective {} != 2",
        report.solution.objective
    );
    assert_eq!(report.solution.solve_result_num, 0, "AMPL srn 0 = solved");
    assert_eq!(report.problem.n_variables, 2);
    assert_eq!(report.problem.n_constraints, 1);
    assert!(report.problem.minimize);
    // The convex IPM ran at least one iteration and recorded it.
    assert!(
        report.statistics.iteration_count >= 1,
        "iteration_count = {}",
        report.statistics.iteration_count
    );
    // Real final KKT residuals (recomputed from the solution), tiny at the
    // optimum — not the placeholder zeros.
    assert!(
        report.statistics.final_constr_viol < 1e-6,
        "constr_viol = {}",
        report.statistics.final_constr_viol
    );
    assert!(
        report.statistics.final_dual_inf < 1e-6,
        "dual_inf = {}",
        report.statistics.final_dual_inf
    );
    assert!(
        report.statistics.final_kkt_error < 1e-6,
        "kkt_error = {}",
        report.statistics.final_kkt_error
    );
    // FAIR provenance is present (solver name, license).
    assert!(!report.fair_metadata.solver.name.is_empty());
}

/// At `--json-detail full` the convex-QP report carries the per-iteration
/// convergence trace (the `iterations` array), the same schema the NLP path
/// uses — so the benchmark harness gets per-iteration data for QP solves too.
#[test]
fn qp_full_report_has_iteration_trace() {
    let dir = std::env::temp_dir();
    let json = dir.join("pounce_convex_qp_full.json");
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("--json-detail")
        .arg("full")
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));

    let text = std::fs::read_to_string(&json).expect("report written");
    let report: SolveReport = serde_json::from_str(&text).expect("deserialize");
    assert!(
        !report.iterations.is_empty(),
        "full-detail QP report should carry an iteration trace"
    );
    // Iteration indices are 0-based and contiguous; the last iterate is the
    // (near-)optimal one.
    for (k, rec) in report.iterations.iter().enumerate() {
        assert_eq!(rec.iter as usize, k, "iteration indices contiguous");
    }
    let last = report.iterations.last().unwrap();
    assert!(
        (last.objective - 2.0).abs() < 1e-4,
        "final traced objective {} ~ 2",
        last.objective
    );
}

/// The `qp_presolve` option toggles presolve on the convex path; both
/// settings must solve the fixture to the same optimum.
#[test]
fn qp_presolve_option_on_and_off_agree() {
    let run = |presolve: &str| -> i32 {
        let out = Command::new(pounce_exe())
            .arg(fixture())
            .arg("--no-sol")
            .arg("solver_selection=qp-ipm")
            .arg(format!("qp_presolve={presolve}"))
            .output()
            .expect("spawn pounce");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("Optimal Solution Found"),
            "qp_presolve={presolve} should solve"
        );
        out.status.code().unwrap_or(-1)
    };
    assert_eq!(run("yes"), 0);
    assert_eq!(run("no"), 0);
}

/// Regression (issue #133): the active-set QP path must solve the NETLIB
/// `afiro` LP under the **default** anti-cycling rule. Previously the
/// steepest-violation default cycled in elastic phase-1 and bailed at
/// iteration 0 with "Search Direction is becoming Too Small" (objective 0,
/// constraint violation 44), while `sqp_qp_anti_cycling=bland` solved it.
/// Phase-1 now uses Bland's provably-finite rule internally, so the default
/// path reaches the true optimum (-464.7531...). Both anti-cycling settings
/// must agree.
#[test]
fn afiro_active_set_solves_under_default_anti_cycling() {
    const AFIRO_OPT: f64 = -4.6475314286e+02;
    let run = |extra: Option<&str>| -> SolveReport {
        let dir = std::env::temp_dir();
        let json = dir.join(format!(
            "pounce_afiro_{}.json",
            extra.unwrap_or("default").replace(['=', ' '], "_")
        ));
        let _ = std::fs::remove_file(&json);
        let mut cmd = Command::new(pounce_exe());
        cmd.arg(fixture_named("lp_afiro.nl"))
            .arg("--no-sol")
            .arg("--json-output")
            .arg(&json)
            .arg("solver_selection=qp-active-set");
        if let Some(e) = extra {
            cmd.arg(e);
        }
        let out = cmd.output().expect("spawn pounce");
        assert_eq!(
            out.status.code(),
            Some(0),
            "afiro qp-active-set ({}) should exit 0; stdout=\n{}",
            extra.unwrap_or("default"),
            String::from_utf8_lossy(&out.stdout)
        );
        let text = std::fs::read_to_string(&json).expect("JSON report written");
        serde_json::from_str(&text).expect("deserialize report")
    };

    // Default rule (the issue's failing case) now reaches the optimum.
    let def = run(None);
    assert_eq!(def.solution.solve_result_num, 0, "afiro default = solved");
    assert!(
        (def.solution.objective - AFIRO_OPT).abs() < 1e-4,
        "afiro default objective {} != {AFIRO_OPT}",
        def.solution.objective
    );

    // Explicit Bland agrees (it always solved afiro).
    let bland = run(Some("sqp_qp_anti_cycling=bland"));
    assert!(
        (bland.solution.objective - AFIRO_OPT).abs() < 1e-4,
        "afiro bland objective {} != {AFIRO_OPT}",
        bland.solution.objective
    );
}
