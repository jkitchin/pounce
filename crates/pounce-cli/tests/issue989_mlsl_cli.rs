//! gh#989 (remaining item f): `pounce --minima mlsl` at parity with
//! `pounce.find_minima(method="mlsl")`. The default `gamma` was 2, which on a
//! 2-D box gave a critical radius comparable to the box diagonal: the
//! six-hump camel launched 2 solves before the sample cap, reported as
//! `budget_exhausted`. Default is now 0.5, and the sample cap reports
//! `sample_cap_reached`.

use std::path::PathBuf;
use std::process::Command;

/// Six-hump camel, `4x² − 2.1x⁴ + x⁶/3 + xy − 4y² + 4y⁴` on [-3,3]×[-2,2];
/// six local minima.
const CAMEL: &str = "g3 1 1 0\n 2 0 1 0 0\n 0 1\n 0 0\n 0 2 0\n 0 0 0 1\n 0 0 0 0 0\n 0 2\n 0 0\n 0 0 0 0 0\nO0 0\no54\n6\no2\nn4\no5\nv0\nn2\no2\nn-2.1\no5\nv0\nn4\no3\no5\nv0\nn6\nn3\no2\nv0\nv1\no2\nn-4\no5\nv1\nn2\no2\nn4\no5\nv1\nn4\nx2\n0 0.5\n1 0.5\nb\n0 -3 3\n0 -2 2\nk1\n0\nG0 2\n0 0\n1 0\n";

fn run(tag: &str, extra: &[&str]) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!("pounce_989f_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let nl = dir.join("camel.nl");
    std::fs::write(&nl, CAMEL).unwrap();
    let json = dir.join("out.json");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .args(["--minima", "mlsl", "--n-minima", "6", "--patience", "15"])
        .args(["--dedup", "1e-3", "--seed", "0"])
        .args(extra)
        .arg("--json-output")
        .arg(&json)
        .arg("--no-sol")
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|_| {
        panic!(
            "no report; stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn status_and_count(r: &serde_json::Value) -> (String, usize) {
    let m = &r["minima"];
    let status = m["status"].as_str().unwrap_or_default().to_string();
    let count = m["minima"].as_array().map_or(0, |a| a.len());
    (status, count)
}

#[test]
fn mlsl_default_gamma_finds_all_six_camel_minima() {
    let r = run("default", &["--max-solves", "60"]);
    let (status, count) = status_and_count(&r);
    assert_eq!(
        status, "target_reached",
        "report minima block: {}",
        r["minima"]
    );
    assert_eq!(count, 6, "report minima block: {}", r["minima"]);
}

#[test]
fn the_sample_cap_is_named_as_such() {
    // A huge critical radius rejects every sample, so no solve is spent and
    // only the sample cap can stop the run.
    let r = run(
        "cap",
        &[
            "--max-solves",
            "5",
            "--gamma",
            "100",
            "--samples-per-round",
            "4",
        ],
    );
    let (status, _) = status_and_count(&r);
    assert_eq!(
        status, "sample_cap_reached",
        "report minima block: {}",
        r["minima"]
    );
}
