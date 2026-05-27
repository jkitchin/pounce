//! Reverse propagation through an [`FbbtTape`] (issue [#62]).
//!
//! Given the per-slot interval bag produced by the forward pass and a
//! target interval on the *root* slot (the constraint's `[g_lb,
//! g_ub]`), the reverse pass walks the tape backwards. For each op,
//! we ask: "given the parent's tightened interval and each
//! operand's current forward interval, what tighter interval can
//! each operand have?" The result is a tightened per-slot interval
//! bag, from which the orchestrator reads back the tightened
//! variable bounds.
//!
//! ## Per-operator inverse rules
//!
//! Each rule below is sound — it returns intervals that contain
//! every feasible operand value, so we never drop a feasible point.
//! Some operators (sin, cos) don't have a tractable interval
//! inverse; the rules for those leave the operand unchanged
//! ("decline to tighten").
//!
//! See Belotti, Cafieri, Lee, Liberti (2010), §3, for the canonical
//! list. The implementations here intersect the inverse with the
//! current forward interval, which is the standard FBBT step.
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62
//! [`FbbtTape`]: pounce_nlp::FbbtTape

use pounce_common::types::Number;
use pounce_nlp::expression_provider::{FbbtOp, FbbtTape};

use crate::fbbt::interval::Interval;

/// Result of [`reverse_pass`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReverseResult {
    /// Per-slot tightened interval. Same length as `tape.ops`. Entry
    /// `i` is the intersection of the forward interval with whatever
    /// constraints reverse-propagation pushed onto slot `i`.
    pub slots: Vec<Interval>,
    /// `true` if the root interval intersected with the constraint
    /// bound was empty — i.e. FBBT detected that **this constraint
    /// is infeasible at the current variable box**. The orchestrator
    /// flags this back to the caller as a presolve-detected
    /// infeasibility; downstream slots are irrelevant in that case.
    pub infeasible: bool,
}

/// Walk `tape` in reverse, propagating the constraint bound
/// `con_bound` (the `[g_lb, g_ub]` of the constraint this tape
/// represents) into each slot. Returns the per-slot tightened
/// intervals.
///
/// The forward pass MUST have been run first (`forward.len() ==
/// tape.ops.len()`); we do not recompute it here.
pub fn reverse_pass(
    tape: &FbbtTape,
    forward: &[Interval],
    con_bound: Interval,
) -> ReverseResult {
    assert_eq!(
        forward.len(),
        tape.ops.len(),
        "forward bag length must match tape"
    );
    if tape.ops.is_empty() {
        return ReverseResult {
            slots: Vec::new(),
            infeasible: con_bound.is_empty(),
        };
    }

    let mut slots = forward.to_vec();
    // Seed: intersect root with the constraint's bound.
    let root_idx = slots.len() - 1;
    let new_root = slots[root_idx].intersect(con_bound);
    if new_root.is_empty() {
        return ReverseResult {
            slots,
            infeasible: true,
        };
    }
    slots[root_idx] = new_root;

    // Walk backward.
    for i in (0..tape.ops.len()).rev() {
        let parent = slots[i];
        if parent.is_empty() {
            // Infeasible somewhere; no point pushing further.
            return ReverseResult {
                slots,
                infeasible: true,
            };
        }
        apply_inverse(&tape.ops[i], parent, &mut slots);
    }
    ReverseResult {
        slots,
        infeasible: false,
    }
}

/// Push the parent's tightened interval back into the operand slots
/// per the inverse rule for `op`. Mutates `slots` in place.
fn apply_inverse(op: &FbbtOp, parent: Interval, slots: &mut [Interval]) {
    match *op {
        FbbtOp::Const(_) | FbbtOp::Var(_) | FbbtOp::Opaque => {
            // Leaves and Opaque: nothing to push into.
        }
        FbbtOp::Add(a, b) => {
            // a + b = z → a ⊆ z - b, b ⊆ z - a.
            let ai = slots[a];
            let bi = slots[b];
            slots[a] = ai.intersect(parent.sub(bi));
            // Recompute the "b ⊆ z - a" arm with the freshly
            // tightened ai (Gauss-Seidel-style FBBT — Belotti §3.2).
            slots[b] = bi.intersect(parent.sub(slots[a]));
        }
        FbbtOp::Sub(a, b) => {
            // a - b = z → a ⊆ z + b, b ⊆ a - z.
            let ai = slots[a];
            let bi = slots[b];
            slots[a] = ai.intersect(parent.add(bi));
            slots[b] = bi.intersect(slots[a].sub(parent));
        }
        FbbtOp::Mul(a, b) => {
            // a * b = z → a ⊆ z / b (when 0 ∉ b), b ⊆ z / a.
            let ai = slots[a];
            let bi = slots[b];
            if !bi.contains_zero() {
                slots[a] = ai.intersect(parent.div(bi));
            }
            // Use the (possibly) tightened a.
            let ai2 = slots[a];
            if !ai2.contains_zero() {
                slots[b] = bi.intersect(parent.div(ai2));
            }
        }
        FbbtOp::Div(a, b) => {
            // a / b = z → a ⊆ z * b. The inverse for b is only
            // useful when 0 ∉ z, since `b ⊆ a / z` requires a
            // divisor disjoint from zero — same condition we already
            // imposed on the forward Div, modulo signs.
            let ai = slots[a];
            let bi = slots[b];
            slots[a] = ai.intersect(parent.mul(bi));
            if !parent.contains_zero() {
                slots[b] = bi.intersect(slots[a].div(parent));
            }
        }
        FbbtOp::Neg(a) => {
            let ai = slots[a];
            slots[a] = ai.intersect(parent.neg());
        }
        FbbtOp::Sqrt(a) => {
            // sqrt(a) = z, z ≥ 0 → a ⊆ z².
            let ai = slots[a];
            let z_pos = parent.intersect(Interval::new(0.0, Number::INFINITY));
            if z_pos.is_empty() {
                slots[a] = Interval::EMPTY;
            } else {
                slots[a] = ai.intersect(z_pos.pow_uint(2));
            }
        }
        FbbtOp::Exp(a) => {
            // exp(a) = z, z > 0 → a ⊆ ln(z).
            let ai = slots[a];
            let z_pos = parent.intersect(Interval::new(0.0, Number::INFINITY));
            if z_pos.is_empty() || z_pos.hi <= 0.0 {
                slots[a] = Interval::EMPTY;
            } else {
                slots[a] = ai.intersect(z_pos.ln());
            }
        }
        FbbtOp::Ln(a) => {
            // ln(a) = z → a ⊆ exp(z).
            let ai = slots[a];
            slots[a] = ai.intersect(parent.exp());
        }
        FbbtOp::Abs(a) => {
            // |a| = z, z ⊆ [0, ∞] → a ⊆ [-z.hi, z.hi].
            let ai = slots[a];
            let z_nonneg = parent.intersect(Interval::new(0.0, Number::INFINITY));
            if z_nonneg.is_empty() {
                slots[a] = Interval::EMPTY;
            } else {
                let envelope = Interval::new(-z_nonneg.hi, z_nonneg.hi);
                slots[a] = ai.intersect(envelope);
            }
        }
        FbbtOp::PowInt(a, n) => {
            let ai = slots[a];
            slots[a] = ai.intersect(inverse_powint(parent, n, ai));
        }
        FbbtOp::Sin(_) | FbbtOp::Cos(_) => {
            // Periodic, multi-branch inverse — defer (no tightening).
        }
    }
}

/// `a^n = z` → tightened envelope on `a`, intersected against the
/// *prior* interval for `a` (so we get the correct branch when `n`
/// is even). Returns the envelope (an interval to intersect with the
/// current operand value).
fn inverse_powint(z: Interval, n: u32, prior_a: Interval) -> Interval {
    if z.is_empty() {
        return Interval::EMPTY;
    }
    if n == 0 {
        // a^0 = 1 — the constraint cannot tell us anything about a.
        return Interval::ENTIRE;
    }
    if n == 1 {
        return z;
    }
    if n % 2 == 1 {
        // Odd: real-valued cube/quintic/... root is monotone.
        let lo = signed_nth_root(z.lo, n);
        let hi = signed_nth_root(z.hi, n);
        Interval::new(lo, hi)
    } else {
        // Even: z must be non-negative.
        let z_pos = z.intersect(Interval::new(0.0, Number::INFINITY));
        if z_pos.is_empty() {
            return Interval::EMPTY;
        }
        // |a| ∈ [sqrt(z.lo), sqrt(z.hi)] (with `^(1/n)` for general
        // even n).
        let abs_lo = z_pos.lo.powf(1.0 / n as f64);
        let abs_hi = z_pos.hi.powf(1.0 / n as f64);
        // Two branches: a ∈ [-abs_hi, -abs_lo] ∪ [abs_lo, abs_hi].
        // We can't return a union, so pick the branch that
        // intersects `prior_a` (the orchestrator-typical case). If
        // both branches intersect, fall back to the convex hull
        // [-abs_hi, abs_hi].
        let pos_branch = Interval::new(abs_lo, abs_hi);
        let neg_branch = Interval::new(-abs_hi, -abs_lo);
        let pos_hit = !prior_a.intersect(pos_branch).is_empty();
        let neg_hit = !prior_a.intersect(neg_branch).is_empty();
        match (pos_hit, neg_hit) {
            (true, false) => pos_branch,
            (false, true) => neg_branch,
            // Both branches feasible — return their hull (the
            // smallest single interval containing both).
            (true, true) => Interval::new(-abs_hi, abs_hi),
            // Neither branch hits — operand is empty.
            (false, false) => Interval::EMPTY,
        }
    }
}

/// `signum(x) * |x|^(1/n)` — the real-valued nth root for odd `n`
/// (defined on the whole real line). Returns `±∞` unchanged.
fn signed_nth_root(x: Number, n: u32) -> Number {
    if !x.is_finite() {
        return x;
    }
    let mag = x.abs().powf(1.0 / n as f64);
    if x < 0.0 {
        -mag
    } else {
        mag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tape: &FbbtTape, forward: &[Interval], bound: Interval) -> ReverseResult {
        reverse_pass(tape, forward, bound)
    }

    /// `x + 1 ∈ [2, 4]` ⇒ `x ⊆ [1, 3]`.
    #[test]
    fn add_constant_tightens() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Const(1.0), FbbtOp::Add(0, 1)],
        };
        let forward = vec![
            Interval::new(-10.0, 10.0),
            Interval::point(1.0),
            Interval::new(-9.0, 11.0),
        ];
        let bound = Interval::new(2.0, 4.0);
        let r = run(&tape, &forward, bound);
        assert!(!r.infeasible);
        // Slot 0 (Var(0)) must be tightened to [1, 3].
        let v0 = r.slots[0];
        assert!(v0.lo >= 1.0 - 1e-12, "v0.lo = {}", v0.lo);
        assert!(v0.hi <= 3.0 + 1e-12, "v0.hi = {}", v0.hi);
    }

    /// `2 * x ∈ [4, 10]` ⇒ `x ⊆ [2, 5]`.
    #[test]
    fn mul_constant_tightens() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Const(2.0), FbbtOp::Var(0), FbbtOp::Mul(0, 1)],
        };
        let forward = vec![
            Interval::point(2.0),
            Interval::new(-100.0, 100.0),
            Interval::new(-200.0, 200.0),
        ];
        let bound = Interval::new(4.0, 10.0);
        let r = run(&tape, &forward, bound);
        assert!(!r.infeasible);
        let v1 = r.slots[1];
        assert!(v1.lo >= 2.0 - 1e-12);
        assert!(v1.hi <= 5.0 + 1e-12);
    }

    /// `x² ∈ [4, 9]` with `x ∈ [-10, 0]` ⇒ `x ⊆ [-3, -2]`.
    #[test]
    fn even_pow_picks_negative_branch() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 2)],
        };
        // Forward: x ∈ [-10, 0] → x² ∈ [0, 100].
        let forward = vec![Interval::new(-10.0, 0.0), Interval::new(0.0, 100.0)];
        let r = run(&tape, &forward, Interval::new(4.0, 9.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= -3.0 - 1e-9, "got {}", v0.lo);
        assert!(v0.hi <= -2.0 + 1e-9, "got {}", v0.hi);
    }

    /// `x³ ∈ [-8, 27]` with `x ∈ [-100, 100]` ⇒ `x ⊆ [-2, 3]`.
    #[test]
    fn odd_pow_inverts_monotonically() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 3)],
        };
        let forward = vec![
            Interval::new(-100.0, 100.0),
            Interval::new(-1e6, 1e6),
        ];
        let r = run(&tape, &forward, Interval::new(-8.0, 27.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= -2.0 - 1e-9, "got {}", v0.lo);
        assert!(v0.hi <= 3.0 + 1e-9, "got {}", v0.hi);
    }

    /// `sqrt(x) ∈ [1, 2]` ⇒ `x ⊆ [1, 4]`.
    #[test]
    fn sqrt_inverse() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Sqrt(0)],
        };
        let forward = vec![Interval::new(-10.0, 100.0), Interval::new(0.0, 10.0)];
        let r = run(&tape, &forward, Interval::new(1.0, 2.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= 1.0 - 1e-12);
        assert!(v0.hi <= 4.0 + 1e-12);
    }

    /// `exp(x) ∈ [1, e]` ⇒ `x ⊆ [0, 1]`.
    #[test]
    fn exp_inverse() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Exp(0)],
        };
        let forward = vec![Interval::new(-10.0, 10.0), Interval::new(0.0, 1.0e5)];
        let r = run(&tape, &forward, Interval::new(1.0, std::f64::consts::E));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= 0.0 - 1e-12);
        assert!(v0.hi <= 1.0 + 1e-12);
    }

    /// `ln(x) ∈ [0, 1]` ⇒ `x ⊆ [1, e]`.
    #[test]
    fn ln_inverse() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Ln(0)],
        };
        let forward = vec![Interval::new(0.5, 100.0), Interval::new(-1.0, 5.0)];
        let r = run(&tape, &forward, Interval::new(0.0, 1.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= 1.0 - 1e-12);
        assert!(v0.hi <= std::f64::consts::E + 1e-12);
    }

    /// `|x| ∈ [0, 2]` with `x ∈ [-10, 10]` ⇒ `x ⊆ [-2, 2]`.
    #[test]
    fn abs_inverse_envelope() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Abs(0)],
        };
        let forward = vec![Interval::new(-10.0, 10.0), Interval::new(0.0, 10.0)];
        let r = run(&tape, &forward, Interval::new(0.0, 2.0));
        assert!(!r.infeasible);
        let v0 = r.slots[0];
        assert!(v0.lo >= -2.0 - 1e-12);
        assert!(v0.hi <= 2.0 + 1e-12);
    }

    /// `(x + y) ∈ [1, 1]` with `x, y ∈ [0, 1]` ⇒ both tighten to
    /// `[0, 1]`. Already at the box; reverse pass shouldn't widen.
    #[test]
    fn add_already_tight_does_not_widen() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Var(1), FbbtOp::Add(0, 1)],
        };
        let forward = vec![
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 2.0),
        ];
        let r = run(&tape, &forward, Interval::point(1.0));
        assert!(!r.infeasible);
        assert!(r.slots[0].lo >= 0.0 && r.slots[0].hi <= 1.0);
        assert!(r.slots[1].lo >= 0.0 && r.slots[1].hi <= 1.0);
    }

    /// Infeasible: `x ∈ [10, 20]` but constraint says `x ∈ [1, 5]`.
    #[test]
    fn root_disjoint_from_bound_is_infeasible() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0)],
        };
        let forward = vec![Interval::new(10.0, 20.0)];
        let r = run(&tape, &forward, Interval::new(1.0, 5.0));
        assert!(r.infeasible);
    }

    /// Opaque slot blocks tightening.
    #[test]
    fn opaque_does_not_propagate() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Opaque, FbbtOp::Add(0, 1)],
        };
        let forward = vec![
            Interval::new(0.0, 10.0),
            Interval::ENTIRE,
            Interval::ENTIRE,
        ];
        let r = run(&tape, &forward, Interval::new(5.0, 5.0));
        assert!(!r.infeasible);
        // Slot 0 still gets some info: x + (anything) = 5 → x ⊆ ?
        // Since opaque is ENTIRE, x is unconstrained — slot 0 stays
        // [0, 10] (the forward bound).
        assert_eq!(r.slots[0], Interval::new(0.0, 10.0));
    }

    /// Soundness fuzz: tighten and resample. Every sample that
    /// satisfies the constraint at the *original* box must still lie
    /// inside the *tightened* per-variable interval. (i.e. FBBT
    /// can't drop a feasible point.)
    #[test]
    fn fuzz_no_overtightening_quadratic_sum() {
        // (x² + y²) = 5, x ∈ [-3, 3], y ∈ [-3, 3].
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),
                FbbtOp::PowInt(0, 2),
                FbbtOp::Var(1),
                FbbtOp::PowInt(2, 2),
                FbbtOp::Add(1, 3),
            ],
        };
        let forward = crate::fbbt::forward::forward_pass(&tape, &[-3.0, -3.0], &[3.0, 3.0])
            .unwrap();
        let r = run(&tape, &forward, Interval::point(5.0));
        assert!(!r.infeasible);

        // For random (x, y) with x² + y² = 5 (sampled on the unit
        // circle, rescaled by sqrt(5)), check both fall in the
        // tightened envelope.
        let var0 = r.slots[0];
        let var1 = r.slots[2];
        let n_samples = 36;
        for k in 0..n_samples {
            let theta = (k as Number) * std::f64::consts::TAU / (n_samples as Number);
            let x = (5.0_f64).sqrt() * theta.cos();
            let y = (5.0_f64).sqrt() * theta.sin();
            assert!(
                var0.contains(x),
                "x={x:.3} not in {:?} (theta={theta})",
                var0
            );
            assert!(
                var1.contains(y),
                "y={y:.3} not in {:?} (theta={theta})",
                var1
            );
        }
    }
}
