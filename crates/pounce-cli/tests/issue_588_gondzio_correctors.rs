//! gh#588 (Q9) — `qp_gondzio_corr` reaches **both** convex drivers.
//!
//! Q9 moved the Gondzio multiple-centrality-corrector scheme out of the HSDE
//! loop into `pounce_convex::correctors`, gave it a knob, and taught the direct
//! driver (`run_ipm`) to run it too. Three things have to be true, and per
//! gh#551 each has to be shown by driving the real CLI rather than by
//! inspecting a field assignment:
//!
//! 1. **The knob is live on the direct driver** — the one that gained the
//!    scheme. `qp_hsde=no` is that route.
//! 2. **The knob is live on HSDE** — the default route, which has always had
//!    the scheme but never a way to switch it off. If the option did not reach
//!    it, a corrector regression in the field would have no bisection handle.
//! 3. **The default did not move.** The registered default (3) is the value
//!    HSDE hard-coded before this phase, so an unset run must be bit-for-bit
//!    where it was. `scripts/sweep-fixtures.sh` over all 57 fixtures diffed
//!    empty against `d38167a9` at defaults, at `nlp_scaling_method=none` and at
//!    `mu_strategy=adaptive`; the `qp_hsde=no` sweep moved five lines, and the
//!    same sweep with `qp_gondzio_corr=0` diffed empty again — which is what
//!    says those five belong to the correctors and to nothing else.
//!
//! `lp_afiro` is the witness for (2). It is NETLIB `afiro`, a small
//! degenerate LP of exactly the family Gondzio's paper is about. Since gh#1246
//! it no longer separates the two settings on the direct driver, so
//! `lp_share1b` is the witness for (1).

use std::path::PathBuf;
use std::process::Command;

use pounce_solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{name}.nl"));
    p
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh588_{}_{seq}_{tag}.{ext}",
        std::process::id()
    ));
    p
}

/// Solve `model` with `opts` appended verbatim to the command line.
fn solve(model: &str, opts: &[&str]) -> SolveReport {
    let tag = format!("{model}_{}", opts.join("_"));
    let json = tmp_path(&tag, "json");
    // Explicit, so a solved fixture does not drop a `.sol` beside the `.nl`.
    let sol = tmp_path(&tag, "sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("--json-detail")
        .arg("summary")
        .arg("print_level=0");
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no report for {model} @ {opts:?} (exit {:?}, {e}); stderr:\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr),
        )
    });
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

fn iters(r: &SolveReport) -> usize {
    r.statistics.iteration_count as usize
}

/// Claim 1: the correctors reach the direct driver, and they pay there.
///
/// Both halves are asserted, but only the *direction* is — absolute counts are
/// the most platform-sensitive numbers in a sweep. Measured at the shipping
/// commit: 10 iterations with the correctors off, 9 with them on.
///
/// That margin used to be a hundred-plus iterations wide (135 off, low 120s
/// on). gh #689 rescaled this driver's cold
/// start and now solves `afiro` in 10 iterations unaided, so what is left here
/// is a one-iteration difference — enough to prove the knob is *wired* to
/// `run_ipm`, which is all this test claims, but no longer evidence that the
/// scheme pays. The aggregate that carries that claim is in
/// `correctors::ALPHA_MAX` (52 NETLIB LPs, 688 -> 644). Do not re-tune the
/// gate against this fixture; its count is chaotic in the threshold.
///
/// The final KKT error does **not** improve along with the count here any
/// more (2.25e-9 off, 6.72e-8 on). Both are converged and the objectives agree
/// to the tolerance asserted below, but the old claim that the shorter
/// trajectory also lands at a sharper point is not true on this model after
/// #689, so it is not made.
///
/// The witness moved from `afiro` to NETLIB `share1b` in gh#1246. Once presolve
/// stopped handing the IPM the bound sides its rows imply, `afiro` took 10
/// iterations either way, so the knob was still wired but this test could not
/// tell. On `share1b` it is 25 iterations off and 22 on.
#[test]
fn the_direct_driver_runs_the_correctors() {
    let off = solve("lp_share1b", &["qp_hsde=no", "qp_gondzio_corr=0"]);
    let on = solve("lp_share1b", &["qp_hsde=no"]);
    assert_ne!(
        iters(&off),
        iters(&on),
        "qp_gondzio_corr does not reach run_ipm: {} iterations either way",
        iters(&on),
    );
    assert!(
        iters(&on) < iters(&off),
        "the correctors cost the direct driver iterations on share1b: {} -> {}",
        iters(&off),
        iters(&on),
    );
    let scale = off.statistics.final_objective.abs().max(1.0);
    assert!(
        (off.statistics.final_objective - on.statistics.final_objective).abs() <= 1e-6 * scale,
        "the optimum moved: {} -> {}",
        off.statistics.final_objective,
        on.statistics.final_objective,
    );
}

/// Claim 2: the same knob reaches the HSDE driver, which is the default route
/// and the one that has had the scheme all along.
///
/// Measured at the shipping commit: 15 iterations off, 13 on.
#[test]
fn the_hsde_driver_honours_the_same_knob() {
    let off = solve("lp_afiro", &["qp_gondzio_corr=0"]);
    let on = solve("lp_afiro", &[]);
    assert_ne!(
        iters(&off),
        iters(&on),
        "qp_gondzio_corr does not reach solve_conic_hsde: {} iterations \
         either way",
        iters(&on),
    );
}

/// Claim 3: the registered default is the historical hard-coded value, so
/// introducing the option changed nothing for a caller who does not set it.
/// This is the whole safety argument for a trajectory change, and CLAUDE.md is
/// explicit that it has to be demonstrated rather than asserted.
#[test]
fn the_registered_default_reproduces_an_unset_run() {
    for route in [vec![], vec!["qp_hsde=no"]] {
        let mut explicit = route.clone();
        explicit.push("qp_gondzio_corr=3");
        let unset = solve("lp_afiro", &route);
        let three = solve("lp_afiro", &explicit);
        assert_eq!(
            iters(&unset),
            iters(&three),
            "qp_gondzio_corr=3 is not the default on route {route:?}",
        );
        assert_eq!(
            unset.statistics.final_objective, three.statistics.final_objective,
            "qp_gondzio_corr=3 is not bit-identical to unset on route {route:?}",
        );
    }
}
