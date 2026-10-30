//! `detach` means "stop pausing; run to completion" — including at `stop-at` checkpoints. A
//! `stop-at` set earlier in the session used to keep pausing after `detach` (breakpoints, events,
//! watchpoints and the terminal pause were all gated on detach; `stop-at` and `stepi` were not), so
//! a client that had ever used `stop-at kkt` could not let the solve finish.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// min x0 s.t. x0^2 + x1^2 = 1, x0 + x1 >= 0.
const SIMPLE: &str = "g3 1 1 0\n 2 2 1 0 1\n 1 0 0 0 0 0\n 0 0\n 2 0 0\n 0 0 0 1\n 0 0 0 0 0\n 4 1\n 0 0\n 0 0 0 0 0\nC0\no0\no5\nv0\nn2\no5\nv1\nn2\nC1\nn0\nO0 0\nn0\nx2\n0 0.5\n1 0.5\nr\n4 1.0\n2 0.0\nb\n0 -10 10\n0 -10 10\nk1\n2\nJ0 2\n0 0\n1 0\nJ1 2\n0 1\n1 1\nG0 1\n0 1\n";

#[test]
fn detach_stops_stop_at_pauses() {
    let mut nl = std::env::temp_dir();
    nl.push(format!("pounce_detach_stop_at_{}.nl", std::process::id()));
    std::fs::write(&nl, SIMPLE).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&nl)
        .args(["--no-sol", "--debug-json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Pause 1 (iteration 0): arm the checkpoint and continue. Pause 2 (after_search_dir): detach.
    // Anything after that is the bug; EOF then aborts the solve instead of letting it finish.
    {
        let mut stdin = child.stdin.take().unwrap();
        for line in [
            r#"{"cmd":"stop-at","args":["kkt"],"id":1}"#,
            r#"{"cmd":"continue","id":2}"#,
            r#"{"cmd":"detach","id":3}"#,
        ] {
            writeln!(stdin, "{line}").unwrap();
        }
    }
    let mut pauses = Vec::new();
    let mut terminated = None;
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line.unwrap()) else {
            continue;
        };
        match ev["event"].as_str() {
            Some("pause") => pauses.push(ev["checkpoint"].as_str().unwrap_or("").to_string()),
            Some("terminated") => terminated = Some(ev),
            _ => {}
        }
    }
    let _ = child.wait();
    let _ = std::fs::remove_file(&nl);
    assert_eq!(
        pauses,
        ["iter_start", "after_search_dir"],
        "no pause after `detach`"
    );
    let terminated = terminated.expect("terminated event");
    assert_eq!(terminated["status"], "SolveSucceeded", "{terminated}");
}
