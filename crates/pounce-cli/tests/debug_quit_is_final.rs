//! `quit` in the JSON debugger must end the process, not just the solve it
//! was typed at.
//!
//! The terminal checkpoint's action is advisory, and the driver can launch
//! further solves after the one the user quit (restoration's inner solve
//! returns into the outer one; the `mu_strategy_fallback` and
//! second-opinion ladders re-solve). Each re-entered the debugger hook,
//! paused again, and blocked on a stdin the client had stopped writing to —
//! so a client that sent `quit` and kept the pipe open (the pounce-studio
//! MCP server did) waited forever. The fix latches `quit`: every later
//! checkpoint returns `Stop` without pausing.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// Waechter-Biegler (2000) from (-2, 1, 1): enters restoration, so the
/// first terminal pause after `continue` is restoration's inner solve.
const WB: &str = "g3 1 1 0\n 3 2 1 0 2\n 1 0\n 0 0\n 1 0 0\n 0 0 0 1 0\n 0 0 0 0 0\n 4 1\n 0 0\n 0 0 0 0 0\nC0\no5\nv0\nn2.0\nC1\nn0\nO0 0\nn0\nr\n4 1.0\n4 0.5\nb\n3\n2 0.0\n2 0.0\nx3\n0 -2.0\n1 1.0\n2 1.0\nk2\n2\n3\nJ0 2\n0 0.0\n1 -1.0\nJ1 2\n0 1.0\n2 -1.0\nG0 1\n0 1.0\n";

#[test]
fn quit_at_an_inner_terminal_pause_ends_the_process() {
    let mut nl = std::env::temp_dir();
    nl.push(format!("pounce_quit_final_{}.nl", std::process::id()));
    std::fs::write(&nl, WB).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&nl)
        .args(["--no-sol", "--debug-json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Keep stdin open for the whole test: closing it would end the solve by
    // EOF and hide the defect.
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<serde_json::Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                if tx.send(v).is_err() {
                    break;
                }
            }
        }
    });
    let next = |what: &str| -> serde_json::Value {
        rx.recv_timeout(Duration::from_secs(30))
            .unwrap_or_else(|_| panic!("no event while waiting for {what}"))
    };
    let wait_pause = |what: &str| loop {
        let ev = next(what);
        if ev["event"] == "pause" {
            break ev;
        }
    };

    wait_pause("initial pause");
    writeln!(stdin, r#"{{"cmd":"continue","id":1}}"#).unwrap();
    stdin.flush().unwrap();
    let p = wait_pause("terminal pause");
    assert_eq!(p["checkpoint"], "terminated");
    assert_eq!(
        p["final"], false,
        "fixture must stop first at restoration's inner terminal pause: {p}"
    );

    writeln!(stdin, r#"{{"cmd":"quit","id":2}}"#).unwrap();
    stdin.flush().unwrap();
    let terminated = loop {
        let ev = next("`terminated` after quit");
        assert_ne!(
            ev["event"], "pause",
            "debugger paused again after quit: {ev}"
        );
        if ev["event"] == "terminated" {
            break ev;
        }
    };
    assert_eq!(terminated["status"], "UserRequestedStop", "{terminated}");

    // The process exits with stdin still open.
    let mut exited = false;
    for _ in 0..100 {
        if child.try_wait().unwrap().is_some() {
            exited = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    if !exited {
        let _ = child.kill();
    }
    drop(stdin);
    let _ = std::fs::remove_file(&nl);
    assert!(exited, "pounce did not exit after `quit` with stdin open");
}
