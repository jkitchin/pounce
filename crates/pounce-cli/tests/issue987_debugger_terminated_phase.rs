//! gh#987 item 7: the debugger's `terminated` checkpoint also fires when
//! restoration's inner solve returns. The pause event now carries
//! `phase` / `final`, so a client can tell the real end of the solve apart.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// min x1 s.t. x1^2 - x2 - 1 = 0, x1 - x3 - 0.5 = 0, x2, x3 >= 0, start
/// (-2, 1, 1): the Waechter-Biegler-like case that needs restoration.
const WB: &str = "g3 1 1 0\n 3 2 1 0 2\n 1 0\n 0 0\n 1 0 0\n 0 0 0 1 0\n 0 0 0 0 0\n 4 1\n 0 0\n 0 0 0 0 0\nC0\no5\nv0\nn2.0\nC1\nn0\nO0 0\nn0\nr\n4 1.0\n4 0.5\nb\n0 -10.0 10.0\n0 0.0 10.0\n0 0.0 10.0\nx3\n0 -2.0\n1 1.0\n2 1.0\nk2\n2\n3\nJ0 2\n0 0.0\n1 -1.0\nJ1 2\n0 1.0\n2 -1.0\nG0 1\n0 1.0\n";

#[test]
fn only_the_last_terminated_pause_is_final() {
    let mut nl = std::env::temp_dir();
    nl.push(format!("pounce_987_wb_{}.nl", std::process::id()));
    std::fs::write(&nl, WB).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&nl)
        .args(["--no-sol", "--debug-json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut terminated: Vec<(bool, bool)> = Vec::new(); // (in_restoration, final)
    let mut id = 0;
    let mut line = String::new();
    'outer: for _ in 0..400 {
        // read until a pause / exit
        loop {
            line.clear();
            if out.read_line(&mut line).unwrap() == 0 {
                break 'outer;
            }
            let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            match ev["event"].as_str() {
                Some("pause") => {
                    if ev["checkpoint"] == "terminated" {
                        terminated.push((
                            ev["in_restoration"].as_bool().unwrap_or(false),
                            ev["final"].as_bool().expect("pause carries `final`"),
                        ));
                        assert_eq!(
                            ev["phase"],
                            if ev["in_restoration"] == true {
                                "restoration"
                            } else {
                                "main"
                            }
                        );
                    }
                    break;
                }
                Some("exit") | Some("finished") => break 'outer,
                _ => {}
            }
        }
        writeln!(stdin, "{{\"cmd\":\"continue\",\"id\":{id}}}").unwrap();
        stdin.flush().unwrap();
        id += 1;
        // swallow the command result
        loop {
            line.clear();
            if out.read_line(&mut line).unwrap() == 0 {
                break 'outer;
            }
            if line.contains("\"result\"") {
                break;
            }
        }
    }
    drop(stdin);
    let _ = child.wait();
    assert!(
        terminated.len() >= 2,
        "expected inner + final terminated pauses: {terminated:?}"
    );
    let (last, rest) = terminated.split_last().unwrap();
    assert_eq!(
        *last,
        (false, true),
        "the real end must be final: {terminated:?}"
    );
    assert!(
        rest.iter().all(|&(resto, fin)| resto && !fin),
        "inner restoration terminations must not be final: {terminated:?}"
    );
}
