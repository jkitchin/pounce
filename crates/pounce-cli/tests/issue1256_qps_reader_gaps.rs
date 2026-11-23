//! gh#1256: QPS / MPS reader gaps, through the CLI.
//!
//! The case that matters most is NETLIB BLEND's: RHS lines with no set name
//! and two (row, value) pairs, on rows whose names are numbers. The #1253
//! reader took the first row name as the set name, paired the first *value*
//! with the second row name (a number, so it parsed), stored that under a row
//! that does not exist, dropped the trailing field — and returned `Ok` with
//! every right-hand side zero. `pounce blend.mps` solved to 0 instead of
//! −30.8121. The fixture below has the same shape: the old reader solves it to
//! 0, the right answer is 12.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

fn tmp_path(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "pounce_issue1256_{}_{n}_{name}",
        std::process::id()
    ))
}

/// Solve `input`, return (report, stderr).
fn solve(input: &PathBuf) -> (SolveReport, String) {
    let json = tmp_path("report.json");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(input)
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "pounce {} failed:\n{}\n{stderr}",
        input.display(),
        String::from_utf8_lossy(&out.stdout)
    );
    let report: SolveReport =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    let _ = std::fs::remove_file(&json);
    (report, stderr)
}

/// max 3x + 2y  s.t.  x + y ≤ 4 (row `65`),  x + 3y ≤ 6 (row `66`)  →  12.
/// RHS written BLEND's way: no set name, two pairs on one line.
const BLEND_SHAPE: &str = "\
NAME BLENDSHAPE
OBJSENSE
    MAX
ROWS
 N obj
 L 65
 L 66
COLUMNS
    x obj 3 65 1
    x 66 1
    y obj 2 65 1
    y 66 3
RHS
    65 4 66 6
ENDATA
";

#[test]
fn rhs_without_set_name_is_not_silently_dropped() {
    let p = tmp_path("blendshape.mps");
    std::fs::write(&p, BLEND_SHAPE).unwrap();
    let (r, _) = solve(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        (r.solution.objective - 12.0).abs() < 1e-6,
        "objective {} != 12 (RHS dropped?)",
        r.solution.objective
    );
}

#[test]
fn gzipped_mps_solves_and_reader_warnings_reach_stderr() {
    // A duplicate RHS entry: the first is kept (as HiGHS does) and the
    // CLI says so on stderr. Read from `.mps.gz`.
    let text = BLEND_SHAPE.replace("    65 4 66 6\n", "    65 4 66 6\n    66 99\n");
    let p = tmp_path("blendshape.MPS.gz");
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(text.as_bytes()).unwrap();
    std::fs::write(&p, enc.finish().unwrap()).unwrap();
    let (r, stderr) = solve(&p);
    let _ = std::fs::remove_file(&p);
    assert!(
        (r.solution.objective - 12.0).abs() < 1e-6,
        "{}",
        r.solution.objective
    );
    assert!(
        stderr.contains("warning") && stderr.contains("1 duplicate RHS"),
        "stderr: {stderr}"
    );
}
