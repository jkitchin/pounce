//! Regression test for pounce#103: `--minima mlsl` must terminate.
//!
//! MLSL grows a sample pool and runs an O(N²) single-linkage scan each
//! round, starting a local solve only for samples that survive the
//! clustering filter. On a problem with few minima it finds them almost
//! immediately and then every later sample is filtered out — so no solve
//! fires, and (before the fix) neither `--max-solves` nor `--patience`
//! advanced, leaving the loop spinning forever while the pool grew.
//!
//! `bounded-quadratic` is a single-minimum bowl: the perfect stall case.
//! This test runs the real binary with a tight budget and a wall-clock
//! guard; a regression would hang, so the timeout (not a wrong answer) is
//! what fails the test.

use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn pounce_exe() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

#[test]
fn mlsl_terminates_on_single_minimum_problem() {
    // Generous relative to the fixed path (<1 s for a 2-var bowl), tight
    // enough to catch a runaway: the pre-fix loop ran > 150 s and never
    // returned.
    const TIMEOUT: Duration = Duration::from_secs(60);

    let mut child = Command::new(pounce_exe())
        .arg("--problem")
        .arg("bounded-quadratic")
        .arg("--minima")
        .arg("mlsl")
        .arg("--n-minima")
        .arg("6")
        .arg("--max-solves")
        .arg("20")
        .arg("--patience")
        .arg("8")
        .arg("--samples-per-round")
        .arg("20")
        // Discard solver output; we only care that the process terminates.
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pounce");

    let start = Instant::now();
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                // It returned — the loop is bounded. A single-minimum bowl
                // exhausts its target/patience cleanly, so exit is success.
                assert!(
                    status.success(),
                    "pounce --minima mlsl exited unsuccessfully: {status:?}"
                );
                return;
            }
            None => {
                if start.elapsed() > TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!(
                        "pounce#103 regression: `--minima mlsl` did not terminate within {}s \
                         (max-solves 20, patience 8) — the solve-gated loop is spinning again",
                        TIMEOUT.as_secs()
                    );
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}
