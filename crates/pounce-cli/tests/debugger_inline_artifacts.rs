//! `artifacts inline` and `ask --prompt-only`: the two debugger settings a
//! front end with no filesystem and no LLM CLI needs (the browser studio,
//! where `viz` cannot write a file or launch a viewer, and `ask` cannot shell
//! out). On a native build the defaults are unchanged — files, and a real
//! `ask` — and the inline forms are opt-in; the wasm build defaults to inline
//! (`crates/pounce-wasm/tests/debug_protocol.mjs` covers that side).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// min x0 s.t. x0^2 + x1^2 = 1, x0 + x1 >= 0 (pounce-wasm's `tests/simple.nl`).
const SIMPLE: &str = "g3 1 1 0\t# hand-written fixture for the pounce-wasm entry points\n 2 2 1 0 1 \t# vars, constraints, objectives, ranges, eqns\n 1 0 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 2 0 0 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 4 1 \t# nonzeros in Jacobian, obj. gradient\n 0 0\t# max name lengths: constraints, variables\n 0 0 0 0 0\t# common exprs: b,c,o,c1,o1\nC0\no0\no5\nv0\nn2\no5\nv1\nn2\nC1\nn0\nO0 0\nn0\nx2\n0 0.5\n1 0.5\nr\n4 1.0\n2 0.0\nb\n0 -10 10\n0 -10 10\nk1\n2\nJ0 2\n0 0\n1 0\nJ1 2\n0 1\n1 1\nG0 1\n0 1\n";

#[test]
fn inline_artifacts_and_prompt_only_ask_return_their_payloads() {
    let mut nl = std::env::temp_dir();
    nl.push(format!("pounce_inline_artifacts_{}.nl", std::process::id()));
    std::fs::write(&nl, SIMPLE).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&nl)
        .args(["--no-sol", "--debug-json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Queue the whole session: the first pause (iteration 0) takes commands
    // until `step`, the next pause (iteration 1) the rest. `detach` lets the
    // solve run to the end, so EOF afterwards is harmless.
    let script = [
        r#"{"cmd":"artifacts","id":1}"#,
        r#"{"cmd":"artifacts","args":["inline"],"id":2}"#,
        r#"{"cmd":"step","id":3}"#,
        r#"{"cmd":"viz","args":["kkt"],"id":4}"#,
        r#"{"cmd":"viz","args":["L"],"id":5}"#,
        r#"{"cmd":"viz","args":["x"],"id":6}"#,
        r#"{"cmd":"save","id":7}"#,
        r#"{"cmd":"ask","args":["--prompt-only","is","mu","stuck?"],"id":8}"#,
        r#"{"cmd":"artifacts","args":["file"],"id":9}"#,
        r#"{"cmd":"artifacts","id":10}"#,
        r#"{"cmd":"detach","id":11}"#,
    ];
    {
        let mut stdin = child.stdin.take().unwrap();
        for line in script {
            writeln!(stdin, "{line}").unwrap();
        }
    }
    let out = BufReader::new(child.stdout.take().unwrap());
    let mut results: HashMap<u64, serde_json::Value> = HashMap::new();
    let mut hello = None;
    for line in out.lines() {
        let Ok(ev) = serde_json::from_str::<serde_json::Value>(&line.unwrap()) else {
            continue;
        };
        match ev["event"].as_str() {
            Some("hello") => hello = Some(ev),
            Some("result") => {
                if let Some(id) = ev["request_id"].as_u64() {
                    results.insert(id, ev);
                }
            }
            _ => {}
        }
    }
    let status = child.wait().unwrap();
    let _ = std::fs::remove_file(&nl);
    assert!(status.success(), "pounce exited with {status}");

    let hello = hello.expect("hello event");
    assert_eq!(
        hello["capabilities"]["inline_artifacts"], false,
        "native default is files"
    );
    assert_eq!(hello["capabilities"]["ask_prompt_only"], true);
    assert!(
        hello["commands"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "artifacts")
    );

    let data = |id: u64| {
        let r = results
            .get(&id)
            .unwrap_or_else(|| panic!("no result for id {id}"));
        assert_eq!(r["ok"], true, "id {id}: {r}");
        r["data"].clone()
    };
    assert_eq!(data(1)["inline"], false);
    // simple.nl: 2 variables + 1 slack, 2 constraints → 5×5, m = 2 negative.
    let kkt = data(4);
    assert_eq!(kkt["dim"], 5);
    assert_eq!(kkt["n_neg"], kkt["expected_neg"]);
    assert_eq!(kkt["matrix"]["format"], "triplet_1based_lower");
    assert!(kkt.get("path").is_none(), "inline, not a file: {kkt}");
    assert_eq!(data(5)["format"], "strict_lower_1based_permuted");
    assert_eq!(data(6)["values"].as_array().unwrap().len(), 2);
    assert_eq!(data(7)["iterate"]["x"].as_array().unwrap().len(), 2);
    let ask = data(8);
    assert_eq!(ask["question"], "is mu stuck?");
    assert!(ask["prompt"].as_str().unwrap().contains("is mu stuck?"));
    assert!(ask.get("reply").is_none(), "prompt-only never runs an LLM");
    assert_eq!(data(10)["inline"], false, "`artifacts file` switches back");
}
