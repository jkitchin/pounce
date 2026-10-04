//! gh #986 item 1: the CLI aborted with a stack overflow on a left-deep chain
//! of binary `o0` nodes (a legal `.nl` shape — discopt writes one for clnlbeam).
//! The binary now runs on a large-stack thread, and the reader refuses past its
//! own depth guard with a clean error, never an abort.

use std::process::Command;

fn long_sum_nl(n: usize) -> String {
    let mut l: Vec<String> = vec![
        "g3 1 1 0".into(),
        format!(" {n} 1 1 0 1"),
        " 0 1".into(),
        " 0 0".into(),
        format!(" 0 {n} 0"),
        " 0 0 0 1 0".into(),
        " 0 0 0 0 0".into(),
        format!(" {n} {n}"),
        " 0 0".into(),
        " 0 0 0 0 0".into(),
        "C0".into(),
        "n0".into(),
        "O0 0".into(),
    ];
    l.extend(std::iter::repeat_n("o0".to_string(), 2 * n - 1));
    for i in 0..n {
        l.extend([
            "o44".into(),
            format!("v{i}"),
            "o2".into(),
            format!("v{i}"),
            format!("v{i}"),
        ]);
    }
    l.extend(["r".into(), "4 0.5".into(), "b".into()]);
    l.extend((0..n).map(|_| "0 -1.0 1.0".to_string()));
    l.push(format!("k{}", n - 1));
    l.extend((0..n - 1).map(|i| (i + 1).to_string()));
    l.push(format!("J0 {n}"));
    l.extend((0..n).map(|i| format!("{i} 1.0")));
    l.push(format!("G0 {n}"));
    l.extend((0..n).map(|i| format!("{i} 0.0")));
    l.join("\n") + "\n"
}

fn run(n: usize) -> (Option<i32>, String) {
    let dir = std::env::temp_dir().join(format!("gh986_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("long_sum.nl");
    std::fs::write(&path, long_sum_nl(n)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&path)
        .args(["--no-sol", "--no-options-file"])
        .output()
        .expect("run pounce");
    let _ = std::fs::remove_dir_all(&dir);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code(), text)
}

#[test]
fn a_5000_term_o0_chain_solves() {
    let (code, text) = run(5000);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("EXIT: Optimal Solution Found"), "{text}");
}

#[test]
fn a_chain_past_the_reader_guard_is_a_clean_error() {
    let (code, text) = run(50_000);
    assert!(
        code.is_some(),
        "killed by a signal (stack overflow?): {text}"
    );
    assert!(!text.contains("overflowed its stack"), "{text}");
    assert!(text.contains("levels deep"), "{text}");
}
