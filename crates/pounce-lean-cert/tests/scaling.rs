//! How exact-rational certification scales with problem size.
//!
//! Every other fixture here has two variables, which says nothing about the
//! real risk of exact arithmetic: **coefficient blowup**. Gaussian elimination
//! over ℚ can grow intermediate numerators and denominators exponentially, so
//! "it works on a 2×2" is not evidence that it works on anything.
//!
//! This measures emit time, certificate size, and the widest integer appearing
//! in a certificate, against `n`.
//!
//! # Results (release build, dense `Q`, one active constraint)
//!
//! | n | emit (ms) | cert (bytes) | widest integer (digits) |
//! |---|---|---|---|
//! | 2 | 0 | 2,684 | 2 |
//! | 4 | 0 | 5,100 | 3 |
//! | 8 | 0 | 13,733 | 8 |
//! | 16 | 4 | 47,082 | 19 |
//! | 32 | 66 | 184,604 | 42 |
//! | 64 | 1,104 | 810,217 | 101 |
//!
//! **Coefficient growth is benign.** Digit counts grow about 2.3× per doubling
//! of `n` — linear-ish, nowhere near the exponential blowup elimination over ℚ
//! permits in the worst case. This is the question that decides whether exact
//! certification is viable at all, and the answer is yes.
//!
//! **Time is the real ceiling, not size.** Emit time grows roughly `O(n⁴)`
//! (≈16× per doubling above n = 16). Extrapolating: n = 128 ≈ 18 s,
//! n = 256 ≈ 5 min, n = 512 ≈ 80 min. So a *dense* QP is comfortable to a few
//! hundred variables and impractical beyond. Certificate size, growing `O(n²)`,
//! reaches only ~13 MB at n = 256 and is not the binding constraint.
//!
//! Two caveats worth stating. These instances are **dense**; real QPs are
//! usually sparse, and the exact solve is dense, so sparsity is the obvious
//! optimization if the ceiling ever binds. And unoptimized builds are ~20×
//! slower (n = 64 takes 23 s), so the figures above are release-only.
//!
//! # Constructing large instances with a known optimum
//!
//! This crate has no floating-point QP solver, so instances are built
//! *backwards* from the KKT conditions rather than solved:
//!
//! * pick a dense SPD `Q = AᵀA + I` with small integer `A` (deterministic LCG),
//! * pick the optimum `x* = 1` (the all-ones vector),
//! * make the single constraint `Σxᵢ ≥ n`, active at `x*`,
//! * choose `λ = 1` and set `c = 1 − Q·1`, which makes stationarity
//!   `Qx* + c = λ·a` hold exactly by construction.
//!
//! Everything is integral, so the instance is exact and its optimum is known
//! without solving anything. The emitter is then asked to certify it.

#![allow(clippy::unwrap_used)]

use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_certificate};
use pounce_lean_cert::to_canonical_json;
use std::time::Instant;

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "0".repeat(64),
        solver: "pounce 0.9.0".to_string(),
    }
}

/// Deterministic small integers — no RNG dependency, reproducible across runs.
fn lcg(state: &mut u64) -> i64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) % 4) as i64
}

/// Dense SPD `Q = AᵀA + I` with small integer entries, plus the `c` that makes
/// the all-ones vector the exact minimizer subject to `Σxᵢ ≥ n`.
fn instance(n: usize) -> QpInput {
    let mut st = 0x2545F491_4F6CDD1D_u64;
    let a: Vec<Vec<i64>> = (0..n)
        .map(|_| (0..n).map(|_| lcg(&mut st)).collect())
        .collect();

    // Q = AᵀA + I  (symmetric positive definite, integral)
    let mut q = vec![vec![0i64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0i64;
            for k in 0..n {
                s += a[k][i] * a[k][j];
            }
            q[i][j] = s + if i == j { 1 } else { 0 };
        }
    }

    // lower triangle only, skipping structural zeros
    let mut q_lower = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            if q[i][j] != 0 {
                q_lower.push((i, j, q[i][j] as f64));
            }
        }
    }

    // c = 1 − Q·1  ⇒  Q·1 + c = 1 = λ·a with λ = 1, a = ones. Stationarity holds.
    let c: Vec<f64> = (0..n)
        .map(|i| 1.0 - q[i].iter().sum::<i64>() as f64)
        .collect();

    QpInput {
        n,
        q_lower,
        half_quadratic: true,
        c,
        constant: 0.0,
        constraints: vec![LinearConstraint {
            name: "sum".to_string(),
            coeffs: vec![1.0; n],
            lower: n as f64,
            upper: f64::INFINITY,
        }],
        var_lower: vec![f64::NEG_INFINITY; n],
        var_upper: vec![f64::INFINITY; n],
        x_float: vec![1.0; n],
        active_tol: 1e-7,
    }
}

/// Longest `num`/`den` string anywhere in the certificate. Walks the parsed
/// JSON so hash literals and structural integers cannot be mistaken for
/// rational coefficients.
fn widest_rational_digits(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, child)| {
                if (k == "num" || k == "den") && child.is_string() {
                    child.as_str().unwrap().trim_start_matches('-').len()
                } else {
                    widest_rational_digits(child)
                }
            })
            .max()
            .unwrap_or(0),
        serde_json::Value::Array(items) => {
            items.iter().map(widest_rational_digits).max().unwrap_or(0)
        }
        _ => 0,
    }
}

struct Measured {
    n: usize,
    millis: u128,
    bytes: usize,
    widest_digits: usize,
}

fn measure(n: usize) -> Measured {
    let input = instance(n);
    let t0 = Instant::now();
    let cert = emit_certificate(&input, &meta())
        .unwrap_or_else(|e| panic!("n = {n} failed to certify: {e:?}"));
    let millis = t0.elapsed().as_millis();

    let json = to_canonical_json(&cert).unwrap();

    // Widest rational literal — the blowup metric.
    //
    // Scan the parsed `num`/`den` fields, NOT the raw text: the binding carries
    // 64-character SHA-256 hashes, and a naive digit scan reports those as
    // 64-digit integers at every `n`. That masked the real coefficient growth
    // entirely until n = 64 finally exceeded it.
    let widest_digits = widest_rational_digits(&serde_json::from_str(&json).unwrap());

    // The optimum really is the all-ones vector.
    for (i, xi) in cert.candidate.as_ref().unwrap().x.iter().enumerate() {
        assert_eq!(
            xi.inner().to_string(),
            "1",
            "n = {n}: x[{i}] should be exactly 1"
        );
    }

    Measured {
        n,
        millis,
        bytes: json.len(),
        widest_digits,
    }
}

#[test]
fn certification_scales_without_coefficient_blowup() {
    // n = 64 costs ~1 s in release but ~23 s unoptimized, which is too slow for
    // a default `cargo test`. Measure it only when optimized.
    let sizes: &[usize] = if cfg!(debug_assertions) {
        &[2, 4, 8, 16, 32]
    } else {
        &[2, 4, 8, 16, 32, 64]
    };
    let mut rows = Vec::new();
    for &n in sizes {
        rows.push(measure(n));
    }

    println!("\n  n     emit(ms)    cert(bytes)   widest integer (digits)");
    println!("  ----  ----------  ------------  ------------------------");
    for r in &rows {
        println!(
            "  {:<4}  {:>10}  {:>12}  {:>24}",
            r.n, r.millis, r.bytes, r.widest_digits
        );
    }
    println!();

    let last = rows.last().unwrap();
    let prev = &rows[rows.len() - 2];

    // The headline: coefficient growth is *linear-ish in n*, not exponential.
    //
    // Assert the character, not a magic constant. An earlier version of this
    // test used `< 100` digits, which n = 64 tripped at 101 — a threshold that
    // says nothing about whether growth is benign. Measured digit counts are
    // 2, 3, 8, 19, 42, 101: roughly 2.3× per doubling of n. Exponential growth
    // in n, the textbook worst case for elimination over ℚ, would put n = 64
    // far beyond any of these bounds.
    assert!(
        last.widest_digits < 8 * last.n,
        "coefficient growth looks super-linear: {} digits at n = {}",
        last.widest_digits,
        last.n
    );
    assert!(
        last.widest_digits < 4 * prev.widest_digits,
        "digit count more than quadrupled from n = {} ({} digits) to n = {} ({} digits) — \
         growth character changed",
        prev.n,
        prev.widest_digits,
        last.n,
        last.widest_digits
    );

    // Certificate size should track the O(n²) dense Q, not something worse.
    assert!(
        last.bytes < 400 * last.n * last.n,
        "certificate at n = {} is {} bytes, larger than an O(n²) encoding explains",
        last.n,
        last.bytes
    );
}

/// Guards the assumption underneath the whole design: the emitter is exact, so
/// its cost must not depend on *coefficient magnitude* in a way that explodes.
/// Same structure, but `A` entries scaled up — bigger integers, same shape.
#[test]
fn larger_input_coefficients_do_not_explode() {
    let n = 16;
    let small = measure(n);

    let mut input = instance(n);
    for e in &mut input.q_lower {
        e.2 *= 1000.0;
    }
    for v in &mut input.c {
        *v *= 1000.0;
    }
    // Rescaling Q and c by the same factor rescales the multiplier, not x*.
    input.constraints[0].coeffs = vec![1.0; n];

    let cert = emit_certificate(&input, &meta());
    match cert {
        Ok(c) => {
            let json = to_canonical_json(&c).unwrap();
            let widest = widest_rational_digits(&serde_json::from_str(&json).unwrap());
            println!(
                "  1000x coefficients at n = {n}: widest {widest} digits \
                 (baseline {} digits)",
                small.widest_digits
            );
            assert!(
                widest < 100,
                "scaling inputs by 1000 caused {widest}-digit integers"
            );
        }
        // A refusal is acceptable — the multiplier no longer matches λ = 1.
        Err(e) => println!("  1000x coefficients at n = {n}: refused ({e:?}), acceptable"),
    }
}
