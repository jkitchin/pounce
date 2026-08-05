//! #462 — on the convex/conic IPM the debugger must answer the commands it
//! *cannot* implement with a capability error, not a timing hint.
//!
//! `print kkt` used to reply "no KKT factorization yet — stop at
//! `after_search_dir`" on the convex backend, and emitted the same message
//! when already stopped at `after_search_dir`: a self-referential loop, since
//! that backend has no augmented-system inertia at any checkpoint. The
//! documented contract (`docs/src/debugger.md`, capability matrix) is that a
//! command unavailable on the current backend "returns an explicit 'not
//! available for this solver' error (it never silently no-ops)" — which the
//! sibling commands (`print rank`, `diagnose`, `resolve`) already did.
//!
//! Covered here, driven over the JSON protocol against the real convex IPM:
//!   - `print kkt` / `print residuals` / `print active` / `print inactive` /
//!     `viz kkt` / `viz L` are rejected as unavailable on the convex backend,
//!     at the very checkpoint the old hint named,
//!   - `print kkt` still works on the NLP filter-IPM at that checkpoint (the
//!     fix is a capability gate, not a blanket disable),
//!   - `hello.capabilities` — what a JSON client is told to feature-detect
//!     off — agrees with the REPL on both backends.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

/// `min x0² + x1²  s.t.  x0 + x1 = 2` — routed to the convex IPM under
/// `solver_selection=qp-ipm`, solvable by the NLP path under `nlp`.
fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("convex_qp.nl");
    p
}

/// Drive `--debug-json` with `cmds`, stopping at `after_search_dir` first
/// (the checkpoint the old hint pointed at). Returns the `hello` event and
/// the `result` events keyed by the command that produced them.
fn drive(solver: &str, cmds: &[&str]) -> (serde_json::Value, Vec<serde_json::Value>) {
    let mut child = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--debug-json")
        .arg(format!("solver_selection={solver}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pounce --debug-json");

    {
        // Taken (not borrowed) so the pipe is *closed* at the end of this
        // block: after `quit` the debugger parks at the terminal checkpoint
        // and reads again, so an open stdin would hang the child forever.
        let mut stdin = child.stdin.take().expect("stdin");
        for (i, c) in cmds.iter().enumerate() {
            writeln!(
                stdin,
                "{{\"cmd\":{},\"id\":{}}}",
                serde_json::json!(c),
                i + 1
            )
            .expect("write command");
        }
        writeln!(stdin, "{{\"cmd\":\"quit\"}}").expect("write quit");
        stdin.flush().expect("flush");
    }

    let stdout = child.stdout.take().expect("stdout");
    let mut hello = serde_json::Value::Null;
    let mut results = Vec::new();
    for line in BufReader::new(stdout).lines() {
        let line = line.expect("read line");
        let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue; // solver banner / report lines are not protocol events
        };
        match ev.get("event").and_then(|e| e.as_str()) {
            Some("hello") => hello = ev,
            Some("result") => results.push(ev),
            _ => {}
        }
    }
    let _ = child.wait();
    (hello, results)
}

fn result_for<'a>(results: &'a [serde_json::Value], cmd: &str) -> &'a serde_json::Value {
    results
        .iter()
        .find(|r| r.get("command").and_then(|c| c.as_str()) == Some(cmd))
        .unwrap_or_else(|| panic!("no `result` event for `{cmd}`; got {results:#?}"))
}

/// The commands the capability matrix marks ➖ for the convex/conic IPM must
/// fail with the backend message — not with a "stop later" timing hint,
/// which is the loop #462 reported.
#[test]
fn convex_backend_rejects_kkt_commands_as_unavailable() {
    let probes = [
        "print kkt",
        "print residuals",
        "print active",
        "print inactive",
        "viz kkt",
        "viz L",
    ];
    let mut cmds = vec!["stop-at kkt", "step"];
    cmds.extend_from_slice(&probes);
    let (_, results) = drive("qp-ipm", &cmds);

    for probe in probes {
        let r = result_for(&results, probe);
        assert_eq!(
            r.get("ok").and_then(|o| o.as_bool()),
            Some(false),
            "`{probe}` must be an error on the convex backend: {r}"
        );
        let out = r.get("output").and_then(|o| o.as_array()).expect("output");
        let msg = out
            .iter()
            .filter_map(|l| l.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            msg.contains("only available for the NLP solver"),
            "`{probe}` must report a capability error, got: {msg}"
        );
        // The specific regression: no timing hint that names a checkpoint
        // we are already stopped at.
        assert!(
            !msg.contains("after_search_dir") && !msg.contains("yet"),
            "`{probe}` must not answer with a timing hint, got: {msg}"
        );
    }
}

/// The same command at the same checkpoint still works on the NLP
/// filter-IPM: #462 is a capability gate on the convex backend, not a
/// blanket removal.
#[test]
fn nlp_backend_still_prints_kkt_at_after_search_dir() {
    let (_, results) = drive("nlp", &["stop-at kkt", "step", "print kkt"]);
    let r = result_for(&results, "print kkt");
    assert_eq!(
        r.get("ok").and_then(|o| o.as_bool()),
        Some(true),
        "`print kkt` must still work on the NLP backend: {r}"
    );
    assert!(
        r.pointer("/data/dim").and_then(|d| d.as_i64()).is_some(),
        "`print kkt` must report the augmented-system dimension: {r}"
    );
}

/// `hello.capabilities` is what the protocol tells clients to feature-detect
/// off, so it must agree with what the REPL actually does on each backend.
#[test]
fn hello_capabilities_match_the_running_backend() {
    let (convex, _) = drive("qp-ipm", &[]);
    let (nlp, _) = drive("nlp", &[]);

    for (name, expected) in [
        ("kkt_inspect", false),
        ("diagnose", false),
        ("mutate_mu", false),
        ("resolve", false),
    ] {
        assert_eq!(
            convex.pointer(&format!("/capabilities/{name}")),
            Some(&serde_json::json!(expected)),
            "convex `capabilities.{name}` disagrees with the REPL: {convex}"
        );
    }
    assert_eq!(
        convex.pointer("/capabilities/viz"),
        Some(&serde_json::json!(["block", "delta"])),
        "convex must not advertise `viz kkt` / `viz L`: {convex}"
    );
    // The advertised iterate blocks are the ones this backend really has.
    assert_eq!(
        convex.pointer("/blocks/0").and_then(|b| b.as_str()),
        Some("x")
    );
    assert!(
        convex
            .pointer("/blocks")
            .and_then(|b| b.as_array())
            .expect("blocks")
            .iter()
            .any(|b| b.as_str() == Some("z")),
        "convex blocks must include `z`: {convex}"
    );

    // The NLP backend keeps every capability it had.
    for name in ["kkt_inspect", "diagnose", "mutate_mu", "resolve"] {
        assert_eq!(
            nlp.pointer(&format!("/capabilities/{name}")),
            Some(&serde_json::json!(true)),
            "NLP `capabilities.{name}` regressed: {nlp}"
        );
    }
    assert_eq!(
        nlp.pointer("/capabilities/viz"),
        Some(&serde_json::json!(["block", "delta", "kkt", "L"])),
    );
}
