//! `quit` in the JSON debugger must end the process, not just the solve it
//! was typed at; and the pause the debugger marks `final` must really be the
//! end of the run.
//!
//! Stopping at one checkpoint is advisory: the solve's own terminal
//! checkpoint follows, and a `quit` typed inside restoration returns into
//! the outer solve. Each re-entered the debugger hook, paused again, and
//! blocked on a stdin the client had stopped writing to — so a client that
//! sent `quit` and kept the pipe open (the pounce-studio MCP server did)
//! waited forever. The fix latches `quit` in JSON mode: every later
//! checkpoint returns `Stop` without pausing.
//!
//! The application's automatic re-solves (`mu_strategy_fallback`, the ℓ₁
//! fallback, the gh#884 retry, the scale retry) never re-entered the hook,
//! which drives one solve; they ran unobserved *after* the `final: true`
//! pause and could replace the verdict it showed. They stand down while a
//! debugger is attached.
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

/// Under the debugger a capped solve is the whole run: the
/// `mu_strategy_fallback` flip (which fires on `Maximum_Iterations_Exceeded`)
/// must not re-solve after the `final: true` pause. Without the stand-down
/// the JSON report carries a `mu_strategy_fallback` record for a retry the
/// debugger never showed.
#[test]
fn no_unobserved_resolve_after_the_final_pause() {
    let dir = std::env::temp_dir();
    let nl = dir.join(format!("pounce_final_is_final_{}.nl", std::process::id()));
    let report = dir.join(format!("pounce_final_is_final_{}.json", std::process::id()));
    std::fs::write(&nl, WB).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&nl)
        .args(["--no-sol", "--debug-json", "--json-output"])
        .arg(&report)
        .arg("max_iter=4")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut events = Vec::new();
    {
        let mut lines = BufReader::new(stdout).lines().map_while(Result::ok);
        // Answer every pause with `continue` until the run reports its end.
        let mut id = 0;
        for line in lines.by_ref() {
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let is_pause = ev["event"] == "pause";
            let done = ev["event"] == "terminated";
            events.push(ev);
            if done {
                break;
            }
            if is_pause {
                id += 1;
                assert!(id < 50, "too many pauses: {events:?}");
                writeln!(stdin, r#"{{"cmd":"continue","id":{id}}}"#).unwrap();
                stdin.flush().unwrap();
            }
        }
    }
    drop(stdin);
    let _ = child.wait();
    let finals: Vec<_> = events
        .iter()
        .filter(|e| e["event"] == "pause" && e["final"] == true)
        .collect();
    assert_eq!(finals.len(), 1, "exactly one final pause: {events:?}");
    let terminated = events.last().unwrap();
    assert_eq!(
        terminated["status"], "MaximumIterationsExceeded",
        "{terminated}"
    );

    let rep: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
    let _ = std::fs::remove_file(&nl);
    let _ = std::fs::remove_file(&report);
    assert!(
        rep["statistics"]["mu_strategy_fallback"].is_null(),
        "a fallback re-solve ran after the debugger's final pause: {}",
        rep["statistics"]["mu_strategy_fallback"]
    );
}
