//! Per-iteration JSONL trace emitter for the studio (issue #68).
//!
//! Activated by `--dump iterates:summary` or `--dump iterates:full`
//! on the CLI (also accepts the legacy `iterate:` spelling); writes
//! one JSONL line per outer/restoration iteration to the persistent
//! `iterates.jsonl` stream at `<dump_dir>/iterates.jsonl`.
//!
//! Schema is locked by the issue's "Proposed flags" section:
//!
//! ```json
//! {"iter":N,"alpha_pr":..,"alpha_du":..,"tag":"..",
//!  "restoration":false,"active_mask":"<b64>",
//!  "x_norm":..,"slack_norm_inf":..,"slack_norm_2":..}
//! ```
//!
//! `full` adds `"x":[..],"slack":[..]`. Vectors are emitted in their
//! native solver coordinates: `x` is the primal vector and `slack` is
//! the per-constraint signed residual computed from `curr_c` / `curr_d
//! - s`.
//!
//! Layered on top of `DiagnosticsState::append_iterate_line`, which
//! owns the buffered file handle.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::diagnostics::{DiagCategory, DiagnosticsState, IterateVariant};
use pounce_common::types::Number;
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::Vector;
use std::fmt::Write as _;

/// Emit one iterate record for the current iteration if (a) the
/// `Iterate` category is enabled in the diagnostics config and (b)
/// the iter filter matches the current iter. Otherwise no-op.
///
/// Must be called *after* `bump_iter`, at the same logical point as
/// the binary `IterDumper::write_record` (post-init for iter 0,
/// post-`accept_trial_point` for the per-iter case) so the captured
/// `iter` field matches `IpData().iter_count()`.
pub(crate) fn emit_record(diag: &DiagnosticsState, data: &IpoptDataHandle, cq: &IpoptCqHandle) {
    if !diag.want(DiagCategory::Iterate) {
        return;
    }
    let variant = diag.config.iterate_variant;
    let json = match build_record(diag, data, cq, variant) {
        Some(s) => s,
        None => return,
    };
    if let Err(e) = diag.append_iterate_line(&json) {
        tracing::warn!(target: "pounce::diagnostics",
            "iterate_dump: failed to append iterate row to iterates.jsonl: {} — continuing",
            e
        );
    }
}

/// Build the JSON line for one iterate. Returns `None` if `curr` is
/// not yet set (defensive — shouldn't happen at the documented hook
/// sites).
fn build_record(
    diag: &DiagnosticsState,
    data: &IpoptDataHandle,
    cq: &IpoptCqHandle,
    variant: IterateVariant,
) -> Option<String> {
    let iter = diag.current_iter();
    let restoration = diag.in_restoration();
    let (alpha_pr, alpha_du, tag, curr_x) = {
        let d = data.borrow();
        let curr = d.curr.as_ref()?.clone();
        (
            d.info_alpha_primal,
            d.info_alpha_dual,
            d.info_alpha_primal_char,
            curr.x.clone(),
        )
    };

    // Constraint-active bitmap. "Active" here = bound-distance below
    // the current barrier parameter `mu`. Matches "whatever pounce
    // already computes internally" per issue open-question #3 — the
    // IPM's complementarity blocks use this same threshold to gauge
    // proximity to the active set.
    let mu = data.borrow().curr_mu.max(1e-12);
    let (active_mask_b64, slack_norm_inf, slack_norm_2, slack_vec) =
        constraint_active_mask_and_slack(cq, mu);

    let x_norm = vec_inf_norm(&*curr_x);

    let mut out = String::with_capacity(256);
    out.push('{');
    write!(out, "\"iter\":{}", iter).ok()?;
    write!(out, ",\"alpha_pr\":{}", json_f64(alpha_pr)).ok()?;
    write!(out, ",\"alpha_du\":{}", json_f64(alpha_du)).ok()?;
    write!(out, ",\"tag\":\"{}\"", escape_tag(tag)).ok()?;
    write!(
        out,
        ",\"restoration\":{}",
        if restoration { "true" } else { "false" }
    )
    .ok()?;
    write!(out, ",\"active_mask\":\"{}\"", active_mask_b64).ok()?;
    write!(out, ",\"x_norm\":{}", json_f64(x_norm)).ok()?;
    write!(out, ",\"slack_norm_inf\":{}", json_f64(slack_norm_inf)).ok()?;
    write!(out, ",\"slack_norm_2\":{}", json_f64(slack_norm_2)).ok()?;

    if matches!(variant, IterateVariant::Full) {
        // x: full primal vector.
        out.push_str(",\"x\":");
        push_vec_json(&mut out, &*curr_x);
        // slack: per-constraint signed residual vector (g(x) - bound),
        // ordered eq-constraints then ineq-constraints.
        out.push_str(",\"slack\":");
        push_slice_json(&mut out, &slack_vec);
    }

    out.push('}');
    Some(out)
}

/// Compute the constraint-active bitmap, the inf-norm and 2-norm of
/// the slack vector, and (for `full`) the slack vector itself.
///
/// Slack convention (matches issue contract):
/// - For equality constraints: `slack[i] = c_i(x)` (signed residual).
/// - For inequality constraints: `slack[i] = d_i(x) - s_i` (signed
///   residual of the IPM's slack-reformulation; same quantity Ipopt
///   uses as `curr_d_minus_s`).
///
/// Active-set notion: bit i set iff `|slack[i]| <= mu`. For
/// equalities every bit is structurally set, but we encode the bound
/// uniformly so studio readers don't have to special-case.
fn constraint_active_mask_and_slack(
    cq: &IpoptCqHandle,
    tol: Number,
) -> (String, Number, Number, Vec<Number>) {
    let c = cq.borrow().curr_c();
    let d_minus_s = cq.borrow().curr_d_minus_s();
    let n_eq = c.dim() as usize;
    let n_ineq = d_minus_s.dim() as usize;
    let m = n_eq + n_ineq;

    let mut slack = Vec::with_capacity(m);
    extend_vec_values(&mut slack, &*c);
    extend_vec_values(&mut slack, &*d_minus_s);
    debug_assert_eq!(slack.len(), m);

    let inf = slack.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let two = slack.iter().fold(0.0_f64, |acc, &v| acc + v * v).sqrt();

    // Bitmap: m bits, little-endian within each byte (LSB = lowest
    // constraint index in that byte). Matches the example bytes in
    // the issue's "Worked example" block.
    let nbytes = (m + 7) / 8;
    let mut bits = vec![0u8; nbytes];
    for (i, &s) in slack.iter().enumerate() {
        if s.abs() <= tol {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    let b64 = base64_encode(&bits);
    (b64, inf, two, slack)
}

fn extend_vec_values(out: &mut Vec<Number>, v: &dyn Vector) {
    if v.dim() == 0 {
        return;
    }
    if let Some(dense) = v.as_any().downcast_ref::<DenseVector>() {
        // expanded_values() materialises homogeneous backings into a
        // full slice; for non-homogeneous it returns the stored values.
        let xs = dense.expanded_values();
        out.extend_from_slice(&xs);
        return;
    }
    // Defensive copy through a fresh DenseVector if the dyn backing
    // isn't a DenseVector. POUNCE is dense-only in v1 so this branch
    // is rare.
    let mut tmp = v.make_new();
    tmp.copy(v);
    if let Some(dense) = tmp.as_any().downcast_ref::<DenseVector>() {
        out.extend_from_slice(&dense.expanded_values());
        return;
    }
    // Last resort: zeros so caller-side dim invariants hold.
    out.resize(out.len() + v.dim() as usize, 0.0);
}

fn vec_inf_norm(v: &dyn Vector) -> Number {
    if v.dim() == 0 {
        return 0.0;
    }
    v.amax()
}

fn push_vec_json(out: &mut String, v: &dyn Vector) {
    let mut buf = Vec::with_capacity(v.dim() as usize);
    extend_vec_values(&mut buf, v);
    push_slice_json(out, &buf);
}

fn push_slice_json(out: &mut String, xs: &[Number]) {
    out.push('[');
    let mut first = true;
    for x in xs {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&json_f64(*x));
    }
    out.push(']');
}

/// JSON-safe float encoding. JSON has no NaN/Inf, so non-finite values
/// emit as `null` (the convention `serde_json` follows). Avoids the
/// `f64::to_string` "1" → "1.0" coercion concern by relying on Rust's
/// default `{}` formatter which prints `1` for integral floats — that
/// matches the issue's worked-example serialisation.
fn json_f64(x: Number) -> String {
    if x.is_finite() {
        // Use `Debug` so integral values print as "1.0" not "1" —
        // matches the worked example in the issue body.
        format!("{:?}", x)
    } else {
        "null".to_string()
    }
}

fn escape_tag(c: char) -> String {
    match c {
        ' ' | '\0' => String::new(),
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
        c => c.to_string(),
    }
}

/// Standard base64 (RFC 4648 §4) with `+ /` and `=` padding. Inlined
/// to avoid adding a `base64` crate dep for ~40 lines of code.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHA: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(((bytes.len() + 2) / 3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b0 = bytes[i] as u32;
        let b1 = bytes[i + 1] as u32;
        let b2 = bytes[i + 2] as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHA[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3f) as usize] as char);
        out.push(ALPHA[(n & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let b0 = bytes[i] as u32;
        let n = b0 << 16;
        out.push(ALPHA[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3f) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = bytes[i] as u32;
        let b1 = bytes[i + 1] as u32;
        let n = (b0 << 16) | (b1 << 8);
        out.push(ALPHA[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3f) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3f) as usize] as char);
        out.push('=');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip_against_known_vectors() {
        assert_eq!(base64_encode(&[]), "");
        assert_eq!(base64_encode(&[0x66]), "Zg==");
        assert_eq!(base64_encode(&[0x66, 0x6f]), "Zm8=");
        assert_eq!(base64_encode(&[0x66, 0x6f, 0x6f]), "Zm9v");
        // The issue's worked example shows "AAAAAA==" for an all-zero
        // 4-byte bitmap and "//8DAA==" for a 4-byte bitmap with the
        // low 19 bits set.
        assert_eq!(base64_encode(&[0, 0, 0, 0]), "AAAAAA==");
        assert_eq!(base64_encode(&[0xff, 0xff, 0x03, 0x00]), "//8DAA==");
    }

    #[test]
    fn json_f64_finite_handles_special_cases() {
        assert_eq!(json_f64(1.0), "1.0");
        assert_eq!(json_f64(0.0), "0.0");
        assert_eq!(json_f64(-0.5), "-0.5");
        assert_eq!(json_f64(f64::NAN), "null");
        assert_eq!(json_f64(f64::INFINITY), "null");
        assert_eq!(json_f64(f64::NEG_INFINITY), "null");
    }

    #[test]
    fn escape_tag_handles_blank_and_special_chars() {
        assert_eq!(escape_tag(' '), "");
        assert_eq!(escape_tag('\0'), "");
        assert_eq!(escape_tag('f'), "f");
        assert_eq!(escape_tag('"'), "\\\"");
        assert_eq!(escape_tag('\\'), "\\\\");
    }
}
