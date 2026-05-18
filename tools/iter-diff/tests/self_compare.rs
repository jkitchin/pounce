//! Self-compare integration test for the POUNCEIT comparator.
//!
//! Builds a synthetic POUNCEIT v1 file in a tempdir, then runs the
//! `compare` engine on it against itself in every tolerance mode and
//! asserts that the result is empty (= match). This exercises the
//! parser end-to-end without depending on either upstream Ipopt or a
//! constrained POUNCE solve.

// Tests panic on setup failure rather than thread an extra error type
// through every helper — the workspace-wide `unwrap_used`/`expect_used`
// warns are aimed at production code paths, not integration tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use iter_diff::{compare, parse_file, Tolerance, MAGIC};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn write_u32<W: Write>(w: &mut W, v: u32) {
    w.write_all(&v.to_le_bytes()).unwrap();
}

fn write_f64<W: Write>(w: &mut W, v: f64) {
    w.write_all(&v.to_le_bytes()).unwrap();
}

fn write_vec<W: Write>(w: &mut W, vals: &[f64]) {
    write_u32(w, vals.len() as u32);
    for v in vals {
        write_f64(w, *v);
    }
}

/// Synthesise an hs071-shaped 3-record file (n=4, m=2). Vector layout
/// matches FORMAT.md's worked example.
fn write_synthetic(path: &PathBuf) {
    let mut f = File::create(path).unwrap();
    // Header.
    f.write_all(MAGIC).unwrap();
    write_u32(&mut f, 1); // version
    write_u32(&mut f, 4); // n
    write_u32(&mut f, 2); // m
    write_u32(&mut f, 0); // nnz_jac (advisory)
    write_u32(&mut f, 0); // nnz_h (advisory)
    let name = b"hs071";
    write_u32(&mut f, name.len() as u32);
    f.write_all(name).unwrap();
    // 3 records.
    for i in 0..3u32 {
        write_u32(&mut f, i); // iter
        write_u32(&mut f, 0); // status
                              // 14 scalars (mu, tau, alpha_pr, alpha_du, delta_x, delta_s,
                              // delta_c, delta_d, inf_pr, inf_du, constr_viol, dual_inf,
                              // complementarity, f).
        for k in 0..14 {
            // Use values that are non-trivial but reproducible.
            let v = (i as f64 + 1.0) * (k as f64 + 0.5);
            write_f64(&mut f, v);
        }
        // x (4), s (1), y_c (1), y_d (1), z_l (4), z_u (4), v_l (1), v_u (0).
        write_vec(&mut f, &[1.0 + i as f64, 5.0, 5.0, 1.0]);
        write_vec(&mut f, &[2.0 + i as f64]);
        write_vec(&mut f, &[0.5]);
        write_vec(&mut f, &[1.5]);
        write_vec(&mut f, &[1.0, 1.0, 1.0, 1.0]);
        write_vec(&mut f, &[1.0, 1.0, 1.0, 1.0]);
        write_vec(&mut f, &[1.0]);
        write_vec(&mut f, &[]);
        // filter_count = 0
        write_u32(&mut f, 0);
    }
    f.flush().unwrap();
}

fn tmpfile(stem: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "iter_diff_self_compare_{}_{}.bin",
        stem,
        std::process::id()
    ));
    p
}

#[test]
fn self_compare_bit_match() {
    let path = tmpfile("bit");
    write_synthetic(&path);
    let parsed = parse_file(&path).expect("parse");
    let divs = compare(&parsed, &parsed, Tolerance::Bit, false, false);
    let _ = std::fs::remove_file(&path);
    assert!(
        divs.is_empty(),
        "expected bit-exact self-match; got {} divergences: {:?}",
        divs.len(),
        divs.iter().map(|d| format!("{}", d)).collect::<Vec<_>>()
    );
}

#[test]
fn self_compare_ulp_match() {
    let path = tmpfile("ulp");
    write_synthetic(&path);
    let parsed = parse_file(&path).expect("parse");
    let divs = compare(&parsed, &parsed, Tolerance::Ulp(0), false, false);
    let _ = std::fs::remove_file(&path);
    assert!(divs.is_empty(), "expected ulp:0 self-match");
}

#[test]
fn self_compare_strict_filter_match() {
    let path = tmpfile("strict");
    write_synthetic(&path);
    let parsed = parse_file(&path).expect("parse");
    let divs = compare(&parsed, &parsed, Tolerance::Bit, true, false);
    let _ = std::fs::remove_file(&path);
    assert!(
        divs.is_empty(),
        "expected strict self-match; got {:?}",
        divs.iter().map(|d| format!("{}", d)).collect::<Vec<_>>()
    );
}

#[test]
fn divergence_when_one_byte_differs() {
    // Sanity check: hand-corrupt the parsed copy and confirm the
    // comparator flags it.
    let path = tmpfile("corrupt");
    write_synthetic(&path);
    let parsed = parse_file(&path).expect("parse");
    let _ = std::fs::remove_file(&path);
    let mut corrupted = parsed.clone();
    corrupted.1[1].x[0] += 1.0;
    let divs = compare(&parsed, &corrupted, Tolerance::Bit, false, true);
    assert_eq!(divs.len(), 1, "expected exactly one divergence");
}

#[test]
fn parser_round_trip_matches_format() {
    let path = tmpfile("rt");
    write_synthetic(&path);
    let (hdr, records) = parse_file(&path).expect("parse");
    let _ = std::fs::remove_file(&path);
    assert_eq!(hdr.format_version, 1);
    assert_eq!(hdr.n, 4);
    assert_eq!(hdr.m, 2);
    assert_eq!(hdr.name, "hs071");
    assert_eq!(records.len(), 3);
    assert_eq!(records[0].iter, 0);
    assert_eq!(records[1].iter, 1);
    assert_eq!(records[2].iter, 2);
    assert_eq!(records[0].x.len(), 4);
    assert_eq!(records[0].v_u.len(), 0);
}
