//! gh#1182: the convex arm overran `max_wall_time` on large LPs, and once the
//! overrun was long enough the gh #535 LP→NLP fallback handed the model to an
//! NLP arm with no time left. That arm stopped at iteration 0, and the report
//! carried no `x` and no `λ`, although the convex iterate it threw away
//! already met `tol` (MIPLIB `rmatr200-p5`: 42.75 s against
//! `max_wall_time=30`).
//!
//! Three things were wrong. The first test covers the overrun sources
//! together:
//!
//! * **Extraction was `O(n·m)`.** `extract_qp_with_map` built a dense
//!   `vec![0.0; n]` per constraint row and scanned it for nonzeros. On the
//!   generated LP below, at `n = 100 000`, that was 3.4 s before any deadline
//!   existed.
//! * **Presolve ignored the clock.** Up to 32 fixpoint layers ran before the
//!   solver's deadline scope opened, and the aggregation postsolve
//!   attributed leftovers by scanning every column once per survivor
//!   (`O(kept·n)`). Together they took 5.9 s on the same model.
//! * **The fallback ran with the budget spent.** Every branch of that
//!   decision is unit-tested on `convex_qp_next_step` in `main.rs`, because no
//!   deterministic run lands a convex verdict exactly on the deadline. The
//!   second test here is the branch that must not change: an LP that stalls
//!   *before* the deadline still reroutes.
//!
//! Measured on the generated model at `max_wall_time=1`, single thread: 9.7 s
//! of wall clock before, 2.8 s after. Most of what remains is the first KKT
//! factorization, which no deadline check can interrupt.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn temp(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("pounce_1182_{}_{tag}.{ext}", std::process::id()))
}

/// A feasible, bounded, banded LP in `.nl` text format: `n` columns in
/// `[0, 10]`, `m = n/2` equality rows, about five nonzeros per column, each
/// within 20 rows of the column's diagonal position. The right-hand side is
/// `A·x₀` for an interior `x₀`, so the LP is feasible, and the box makes it
/// bounded. The band keeps the KKT factorization's fill small: a uniformly
/// random pattern at this size fills catastrophically, which would measure
/// the factorization rather than the phases this issue is about.
///
/// Deterministic, because it uses its own generator.
fn write_banded_lp(path: &Path, n: usize) {
    let m = n / 2;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    let mut col_count = vec![0usize; n];
    for (j, count) in col_count.iter_mut().enumerate() {
        let base = (j * m / n) as i64;
        let mut picked: Vec<usize> = Vec::with_capacity(5);
        for _ in 0..5 {
            let off = (next() * 41.0) as i64 - 20;
            let i = (base + off).rem_euclid(m as i64) as usize;
            if !picked.contains(&i) {
                picked.push(i);
            }
        }
        for i in picked {
            rows[i].push((j, 2.0 * next() - 1.0));
            *count += 1;
        }
    }
    let x0: Vec<f64> = (0..n).map(|_| 1.0 + 8.0 * next()).collect();
    let c: Vec<f64> = (0..n).map(|_| 2.0 * next() - 1.0).collect();
    let nnz: usize = col_count.iter().sum();

    let mut s = String::with_capacity(48 * nnz);
    use std::fmt::Write as _;
    s.push_str("g3 1 1 0\n");
    let _ = write!(
        s,
        " {n} {m} 1 0 {m} 0\n 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 0 0 0 0 0\n {nnz} {n}\n 0 0\n 0 0 0 0 0\n"
    );
    for i in 0..m {
        let _ = writeln!(s, "C{i}\nn0");
    }
    s.push_str("O0 0\nn0\nr\n");
    for row in &rows {
        let b: f64 = row.iter().map(|&(j, a)| a * x0[j]).sum();
        let _ = writeln!(s, "4 {b:e}");
    }
    s.push_str("b\n");
    for _ in 0..n {
        s.push_str("0 0 10\n");
    }
    let _ = writeln!(s, "k{}", n - 1);
    let mut acc = 0;
    for count in &col_count[..n - 1] {
        acc += count;
        let _ = writeln!(s, "{acc}");
    }
    for (i, row) in rows.iter_mut().enumerate() {
        if row.is_empty() {
            continue;
        }
        row.sort_by_key(|&(j, _)| j);
        let _ = writeln!(s, "J{i} {}", row.len());
        for &(j, a) in row.iter() {
            let _ = writeln!(s, "{j} {a:e}");
        }
    }
    let _ = writeln!(s, "G0 {n}");
    for (j, cj) in c.iter().enumerate() {
        let _ = writeln!(s, "{j} {cj:e}");
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(s.as_bytes()))
        .expect("write generated LP");
}

struct Run {
    stderr: String,
    report: SolveReport,
    wall: Duration,
}

fn solve(model: &Path, tag: &str, opts: &[&str]) -> Run {
    let json = temp(tag, "json");
    let _ = std::fs::remove_file(&json);
    let started = Instant::now();
    let out = Command::new(pounce_exe())
        .arg(model)
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(opts)
        .output()
        .expect("spawn pounce");
    let wall = started.elapsed();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let text = std::fs::read_to_string(&json)
        .unwrap_or_else(|e| panic!("read JSON report ({e}); stdout=\n{stdout}\nstderr=\n{stderr}"));
    let report: SolveReport = serde_json::from_str(&text).expect("parse JSON report");
    let _ = std::fs::remove_file(&json);
    Run {
        stderr,
        report,
        wall,
    }
}

const REROUTE_NOTE: &str = "being re-solved on the general NLP";

/// A large LP whose convex solve the deadline cuts off: the whole run stays
/// near the limit, the fallback does not run, and the report carries a primal
/// and dual vector of the model's shape.
///
/// What this does **not** show is the quality of that point. At this limit the
/// deadline usually crosses before the first iteration, so `x` is the lifted
/// starting point (measured: 1 768 of 100 000 entries nonzero, the columns
/// presolve fixed), not a converged iterate. At `max_wall_time=3` the same
/// model reports a full iterate, but how many iterations fit is
/// machine-dependent, so that is not asserted. `TimeLimit` never rerouted, even
/// before this fix, so this test is about the overrun. The reroute gate is
/// pinned by the `convex_qp_next_step` unit tests.
///
/// The margin is generous on purpose. It allows for one uninterruptible KKT
/// factorization on a slow CI runner, and it still fails on the code before
/// this fix, which spent 9.7 s here on a fast machine before the solver even
/// started.
#[test]
fn a_convex_solve_cut_off_by_the_deadline_stops_near_it() {
    const N: usize = 100_000;
    const LIMIT: f64 = 0.5;
    const MARGIN: Duration = Duration::from_secs(8);
    let model = temp("banded_lp", "nl");
    write_banded_lp(&model, N);
    let limit = format!("max_wall_time={LIMIT}");
    let run = solve(&model, "cutoff", &[&limit]);
    let _ = std::fs::remove_file(&model);

    assert_eq!(
        run.report.solution.engine, "cvx-qp",
        "precondition: `auto` must route this LP to the convex arm"
    );
    assert!(
        run.wall < Duration::from_secs_f64(LIMIT) + MARGIN,
        "the run took {:?} against max_wall_time={LIMIT}; stderr=\n{}",
        run.wall,
        run.stderr
    );
    assert!(
        !run.stderr.contains(REROUTE_NOTE),
        "nothing is left of the budget for a second engine; stderr=\n{}",
        run.stderr
    );
    assert_eq!(
        run.report.solution.status,
        ApplicationReturnStatus::MaximumWallTimeExceeded
    );
    assert_eq!(run.report.solution.x.len(), N, "the convex x is reported");
    assert_eq!(
        run.report.solution.lambda.len(),
        N / 2,
        "the convex constraint multipliers are reported"
    );
    assert!(run.report.solution.x.iter().all(|v| v.is_finite()));
}

/// The other branch: a convex solve that stalls with budget to spare still
/// hands the LP to the NLP arm. `afiro` under an unreachable `tol` is the
/// gh #535 trigger (the convex arm exhausts its iterations in well under a
/// second), and an explicit, generous `max_wall_time` must not turn the
/// fallback off.
///
/// It asserts that the reroute *ran*, not which engine answers. gh#1183
/// (jkitchin/pounce#1184) lets the more accurate of the two candidates
/// report, and on this model that is the convex one, so an `engine == "nlp"`
/// assertion would contradict that PR's test of the same command.
#[test]
fn a_stall_before_the_deadline_still_falls_back() {
    let run = solve(
        &fixture("lp_afiro.nl"),
        "stall",
        &["tol=1e-20", "max_wall_time=300"],
    );
    assert!(
        run.stderr.contains(REROUTE_NOTE),
        "an LP that stalls with budget left must still reroute; stderr=\n{}",
        run.stderr
    );
}
