//! Precompiled symbolic-Hessian program for one `Tape`.
//!
//! `Tape::hessian_accumulate` runs forward-over-reverse AD at every
//! call: for each tape variable `j` it (a) match-dispatches every op
//! in a forward-tangent sweep, (b) zeros adj/adj_dot, (c)
//! match-dispatches every op again in the reverse-over-tangent
//! sweep, and (d) HashMap-looks-up every `Var(k)` slot to find its
//! Hessian output position. On evaluator-bound problems (dirichlet,
//! lane_emden, henon) that match-dispatch + symbolic-AD overhead is
//! ~80% of total CPU.
//!
//! This module compiles all of that ONCE at tape-build time into a
//! flat `Vec<HOp>` of pre-resolved primitive ops:
//!
//!   * Forward pass — one `Fwd*` op per `TapeOp`. Mirrors
//!     `Tape::forward`.
//!   * Per-`j` forward tangent — only the ops touching slots that
//!     statically depend on `j` are emitted (the rest stay zero
//!     from the per-`j` `ZeroRange` reset).
//!   * Per-`j` reverse-over-tangent — only ops on slots reachable
//!     backward from output, with all slot indices and Hessian
//!     output pointers pre-resolved.
//!
//! ## Scratch layout
//!
//! The program reads/writes a single `&mut [f64]` arena of
//! `n_slots` cells. We allocate four contiguous regions of length
//! `n` (`n` = `tape.ops.len()`):
//!
//!   * `v[i]`        in slot `i`
//!   * `dot[i]`      in slot `n + i`
//!   * `adj[i]`      in slot `2n + i`
//!   * `adj_dot[i]`  in slot `3n + i`
//!
//! Per-`j` setup zeros the `[n, 4n)` range and seeds `adj[n-1]`.
//! Allocation pattern is intentionally trivial — finer-grained
//! slot recycling buys little once the dispatch loop is the
//! bottleneck, and a contiguous layout makes the per-`j`
//! `ZeroRange` reset a single `memset`-friendly loop.

use std::collections::HashMap;

use super::nl_tape::{Tape, TapeOp};

/// One primitive operation in the compiled Hessian program.
/// `dst`/`a`/`b`/etc. are `u32` offsets into the caller's scratch
/// slice; see the module docs for the slot layout.
#[derive(Debug, Clone, Copy)]
pub enum HOp {
    // ===== Forward pass =====
    FwdLoadVar {
        dst: u32,
        x_idx: u32,
    },
    FwdLoadConst {
        dst: u32,
        c_idx: u32,
    },
    FwdAdd {
        dst: u32,
        a: u32,
        b: u32,
    },
    FwdSub {
        dst: u32,
        a: u32,
        b: u32,
    },
    FwdMul {
        dst: u32,
        a: u32,
        b: u32,
    },
    FwdDiv {
        dst: u32,
        a: u32,
        b: u32,
    },
    FwdPow {
        dst: u32,
        a: u32,
        b: u32,
    },
    FwdNeg {
        dst: u32,
        a: u32,
    },
    FwdAbs {
        dst: u32,
        a: u32,
    },
    FwdSqrt {
        dst: u32,
        a: u32,
    },
    FwdExp {
        dst: u32,
        a: u32,
    },
    FwdLog {
        dst: u32,
        a: u32,
    },
    FwdLog10 {
        dst: u32,
        a: u32,
    },
    FwdSin {
        dst: u32,
        a: u32,
    },
    FwdCos {
        dst: u32,
        a: u32,
    },

    // ===== Scalar slot init =====
    SetZero {
        dst: u32,
    },
    SetOne {
        dst: u32,
    },

    // ===== Bulk reset (start of each j) =====
    ZeroRange {
        start: u32,
        len: u32,
    },

    // ===== Forward tangent (per j) =====
    DotAdd {
        dst: u32,
        a: u32,
        b: u32,
    },
    DotSub {
        dst: u32,
        a: u32,
        b: u32,
    },
    /// dot[d] = dot[a]*v[b] + v[a]*dot[b]
    DotMul {
        dst: u32,
        dot_a: u32,
        vb: u32,
        va: u32,
        dot_b: u32,
    },
    /// dot[d] = (dot[a]*v[b] - v[a]*dot[b]) / (v[b]*v[b])
    DotDiv {
        dst: u32,
        dot_a: u32,
        vb: u32,
        va: u32,
        dot_b: u32,
    },
    /// dot[d] = 0.5 / v[d] * dot[a]  (v[d] = sqrt(v[a]))
    DotSqrt {
        dst: u32,
        dot_a: u32,
        vd: u32,
    },
    /// dot[d] = v[d] * dot[a]  (v[d] = exp(v[a]))
    DotExp {
        dst: u32,
        dot_a: u32,
        vd: u32,
    },
    DotLog {
        dst: u32,
        dot_a: u32,
        va: u32,
    },
    DotLog10 {
        dst: u32,
        dot_a: u32,
        va: u32,
    },
    DotSin {
        dst: u32,
        dot_a: u32,
        va: u32,
    },
    DotCos {
        dst: u32,
        dot_a: u32,
        va: u32,
    },
    DotNeg {
        dst: u32,
        dot_a: u32,
    },
    DotAbs {
        dst: u32,
        dot_a: u32,
        va: u32,
    },
    /// Compound: dot[d] for Pow(a, b). Carries the runtime
    /// `u != 0` / `u > 0` branches.
    DotPow {
        dst: u32,
        va: u32,
        vb: u32,
        vd: u32,
        dot_a: u32,
        dot_b: u32,
    },

    // ===== Reverse + adj_dot update (per j) =====
    // Each op consumes adj[i] (= `w`) and adj_dot[i] (= `wd`) of
    // the consumer slot, then `+=`-accumulates into the adj /
    // adj_dot of the operand slots.
    RevAdd {
        adj_a: u32,
        adj_b: u32,
        adj_dot_a: u32,
        adj_dot_b: u32,
        w: u32,
        wd: u32,
    },
    RevSub {
        adj_a: u32,
        adj_b: u32,
        adj_dot_a: u32,
        adj_dot_b: u32,
        w: u32,
        wd: u32,
    },
    RevMul {
        adj_a: u32,
        adj_b: u32,
        adj_dot_a: u32,
        adj_dot_b: u32,
        w: u32,
        wd: u32,
        va: u32,
        vb: u32,
        dot_a: u32,
        dot_b: u32,
    },
    RevDiv {
        adj_a: u32,
        adj_b: u32,
        adj_dot_a: u32,
        adj_dot_b: u32,
        w: u32,
        wd: u32,
        va: u32,
        vb: u32,
        dot_a: u32,
        dot_b: u32,
    },
    RevPow {
        adj_a: u32,
        adj_b: u32,
        adj_dot_a: u32,
        adj_dot_b: u32,
        w: u32,
        wd: u32,
        va: u32,
        vb: u32,
        vd: u32,
        dot_a: u32,
        dot_b: u32,
    },
    RevNeg {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
    },
    RevAbs {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        va: u32,
    },
    RevSqrt {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        va: u32,
        vd: u32,
        dot_a: u32,
    },
    RevExp {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        vd: u32,
        dot_a: u32,
    },
    RevLog {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        va: u32,
        dot_a: u32,
    },
    RevLog10 {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        va: u32,
        dot_a: u32,
    },
    RevSin {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        va: u32,
        dot_a: u32,
    },
    RevCos {
        adj_a: u32,
        adj_dot_a: u32,
        w: u32,
        wd: u32,
        va: u32,
        dot_a: u32,
    },

    // ===== Output =====
    /// values[hess_ptr] += weight * scratch[adj_dot_slot].
    HessEmit {
        hess_ptr: u32,
        adj_dot_slot: u32,
    },
}

/// Precompiled Hessian-of-one-tape program. Built once via
/// [`HessianProgram::compile`]; executed many times.
#[derive(Debug, Clone)]
pub struct HessianProgram {
    ops: Vec<HOp>,
    consts: Vec<f64>,
    n_slots: u32,
}

impl HessianProgram {
    /// Build the program. The `hess_map` is the same `(row, col)
    /// -> values-index` map that [`Tape::hessian_accumulate`] uses;
    /// the compiler inlines each lookup into a `HessEmit` op.
    pub fn compile(tape: &Tape, hess_map: &HashMap<(usize, usize), usize>) -> Self {
        let n = tape.ops.len() as u32;
        let v_base = 0u32;
        let dot_base = n;
        let adj_base = 2 * n;
        let adj_dot_base = 3 * n;
        let n_slots = 4 * n;

        let v_slot = |i: u32| v_base + i;
        let dot_slot = |i: u32| dot_base + i;
        let adj_slot = |i: u32| adj_base + i;
        let adj_dot_slot = |i: u32| adj_dot_base + i;

        let reachable = reachable_to_output(tape);
        let var_indices = tape.variables();
        // depends_on[k_idx][i] — does slot i depend on var_indices[k_idx]?
        let depends_on: Vec<Vec<bool>> = (0..var_indices.len())
            .map(|k_idx| depends_on_var(tape, var_indices[k_idx]))
            .collect();

        let mut consts: Vec<f64> = Vec::new();
        let mut const_intern: HashMap<u64, u32> = HashMap::new();
        let mut intern_const = |c: f64, consts: &mut Vec<f64>| -> u32 {
            let bits = c.to_bits();
            if let Some(&idx) = const_intern.get(&bits) {
                return idx;
            }
            let idx = consts.len() as u32;
            consts.push(c);
            const_intern.insert(bits, idx);
            idx
        };

        let mut ops: Vec<HOp> = Vec::new();

        // ---- Forward pass ----
        for (i, tape_op) in tape.ops.iter().enumerate() {
            let i = i as u32;
            let dst = v_slot(i);
            let op = match *tape_op {
                TapeOp::Const(c) => HOp::FwdLoadConst {
                    dst,
                    c_idx: intern_const(c, &mut consts),
                },
                TapeOp::Var(x_idx) => HOp::FwdLoadVar {
                    dst,
                    x_idx: x_idx as u32,
                },
                TapeOp::Add(a, b) => HOp::FwdAdd {
                    dst,
                    a: v_slot(a as u32),
                    b: v_slot(b as u32),
                },
                TapeOp::Sub(a, b) => HOp::FwdSub {
                    dst,
                    a: v_slot(a as u32),
                    b: v_slot(b as u32),
                },
                TapeOp::Mul(a, b) => HOp::FwdMul {
                    dst,
                    a: v_slot(a as u32),
                    b: v_slot(b as u32),
                },
                TapeOp::Div(a, b) => HOp::FwdDiv {
                    dst,
                    a: v_slot(a as u32),
                    b: v_slot(b as u32),
                },
                TapeOp::Pow(a, b) => HOp::FwdPow {
                    dst,
                    a: v_slot(a as u32),
                    b: v_slot(b as u32),
                },
                TapeOp::Neg(a) => HOp::FwdNeg {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Abs(a) => HOp::FwdAbs {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Sqrt(a) => HOp::FwdSqrt {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Exp(a) => HOp::FwdExp {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Log(a) => HOp::FwdLog {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Log10(a) => HOp::FwdLog10 {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Sin(a) => HOp::FwdSin {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Cos(a) => HOp::FwdCos {
                    dst,
                    a: v_slot(a as u32),
                },
                TapeOp::Funcall { .. } => panic!(
                    "HessianProgram path does not support AMPL external functions; \
                     use the Tape (build_with_externals) path instead."
                ),
                TapeOp::Tan(_)
                | TapeOp::Atan(_)
                | TapeOp::Acos(_)
                | TapeOp::Sinh(_)
                | TapeOp::Cosh(_)
                | TapeOp::Tanh(_)
                | TapeOp::Asin(_)
                | TapeOp::Acosh(_)
                | TapeOp::Asinh(_)
                | TapeOp::Atanh(_)
                | TapeOp::Atan2(_, _)
                | TapeOp::Cmp(_, _, _)
                | TapeOp::And(_, _)
                | TapeOp::Or(_, _)
                | TapeOp::Not(_)
                | TapeOp::Select(_, _, _)
                | TapeOp::Min(_, _)
                | TapeOp::Max(_, _) => panic!(
                    "HessianProgram path does not yet support tan/atan/acos, the \
                     other transcendental opcodes, atan2, min/max, or \
                     conditional / logical opcodes; use the Tape \
                     (build_with_externals) interpreter path instead."
                ),
            };
            ops.push(op);
        }

        if n == 0 || var_indices.is_empty() {
            return HessianProgram {
                ops,
                consts,
                n_slots,
            };
        }

        // ---- Per-j forward-tangent + reverse-over-tangent ----
        for (k_idx, &j) in var_indices.iter().enumerate() {
            // Reset dot, adj, adj_dot for this j. Seed adj[n-1] = 1.
            ops.push(HOp::ZeroRange {
                start: dot_base,
                len: 3 * n,
            });
            ops.push(HOp::SetOne {
                dst: adj_slot(n - 1),
            });

            // Forward tangent: only emit ops for slots that
            // statically depend on j (the rest stay zero from the
            // ZeroRange above).
            for (i, tape_op) in tape.ops.iter().enumerate() {
                let i_u = i as u32;
                if !depends_on[k_idx][i] {
                    continue;
                }
                let dst = dot_slot(i_u);
                let dot_op = match *tape_op {
                    // Const: dot stays 0 (filtered above by
                    // depends_on, since Const has no var-deps).
                    TapeOp::Const(_) => continue,
                    // Var(k): dot = 1 iff k == j, else 0. We only
                    // get here if depends_on[k_idx][i] is true,
                    // which for Var(k) means k == j.
                    TapeOp::Var(_) => HOp::SetOne { dst },
                    TapeOp::Add(a, b) => HOp::DotAdd {
                        dst,
                        a: dot_slot(a as u32),
                        b: dot_slot(b as u32),
                    },
                    TapeOp::Sub(a, b) => HOp::DotSub {
                        dst,
                        a: dot_slot(a as u32),
                        b: dot_slot(b as u32),
                    },
                    TapeOp::Mul(a, b) => HOp::DotMul {
                        dst,
                        dot_a: dot_slot(a as u32),
                        vb: v_slot(b as u32),
                        va: v_slot(a as u32),
                        dot_b: dot_slot(b as u32),
                    },
                    TapeOp::Div(a, b) => HOp::DotDiv {
                        dst,
                        dot_a: dot_slot(a as u32),
                        vb: v_slot(b as u32),
                        va: v_slot(a as u32),
                        dot_b: dot_slot(b as u32),
                    },
                    TapeOp::Pow(a, b) => HOp::DotPow {
                        dst,
                        va: v_slot(a as u32),
                        vb: v_slot(b as u32),
                        vd: v_slot(i_u),
                        dot_a: dot_slot(a as u32),
                        dot_b: dot_slot(b as u32),
                    },
                    TapeOp::Neg(a) => HOp::DotNeg {
                        dst,
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Abs(a) => HOp::DotAbs {
                        dst,
                        dot_a: dot_slot(a as u32),
                        va: v_slot(a as u32),
                    },
                    TapeOp::Sqrt(a) => HOp::DotSqrt {
                        dst,
                        dot_a: dot_slot(a as u32),
                        vd: v_slot(i_u),
                    },
                    TapeOp::Exp(a) => HOp::DotExp {
                        dst,
                        dot_a: dot_slot(a as u32),
                        vd: v_slot(i_u),
                    },
                    TapeOp::Log(a) => HOp::DotLog {
                        dst,
                        dot_a: dot_slot(a as u32),
                        va: v_slot(a as u32),
                    },
                    TapeOp::Log10(a) => HOp::DotLog10 {
                        dst,
                        dot_a: dot_slot(a as u32),
                        va: v_slot(a as u32),
                    },
                    TapeOp::Sin(a) => HOp::DotSin {
                        dst,
                        dot_a: dot_slot(a as u32),
                        va: v_slot(a as u32),
                    },
                    TapeOp::Cos(a) => HOp::DotCos {
                        dst,
                        dot_a: dot_slot(a as u32),
                        va: v_slot(a as u32),
                    },
                    TapeOp::Funcall { .. } => panic!(
                        "HessianProgram path does not support AMPL external functions; \
                         use the Tape (build_with_externals) path instead."
                    ),
                    TapeOp::Tan(_)
                    | TapeOp::Atan(_)
                    | TapeOp::Acos(_)
                    | TapeOp::Sinh(_)
                    | TapeOp::Cosh(_)
                    | TapeOp::Tanh(_)
                    | TapeOp::Asin(_)
                    | TapeOp::Acosh(_)
                    | TapeOp::Asinh(_)
                    | TapeOp::Atanh(_)
                    | TapeOp::Atan2(_, _)
                    | TapeOp::Cmp(_, _, _)
                    | TapeOp::And(_, _)
                    | TapeOp::Or(_, _)
                    | TapeOp::Not(_)
                    | TapeOp::Select(_, _, _)
                    | TapeOp::Min(_, _)
                    | TapeOp::Max(_, _) => panic!(
                        "HessianProgram path does not yet support tan/atan/acos, the \
                         other transcendental opcodes, atan2, min/max, or \
                         conditional / logical opcodes; use the Tape \
                         (build_with_externals) interpreter path instead."
                    ),
                };
                ops.push(dot_op);
            }

            // Reverse-over-tangent: walk slots backward, emit only
            // for reachable slots.
            for i in (0..n as usize).rev() {
                if !reachable[i] {
                    continue;
                }
                let i_u = i as u32;
                let w = adj_slot(i_u);
                let wd = adj_dot_slot(i_u);
                let tape_op = &tape.ops[i];
                let rev_op = match *tape_op {
                    TapeOp::Const(_) => continue,
                    TapeOp::Var(k) => {
                        // At a Var slot: if k >= j and hess_map has
                        // an entry for (k, j), emit a HessEmit op.
                        // No adj/adj_dot propagation (no operands).
                        if k >= j {
                            if let Some(&ptr) = hess_map.get(&(k, j)) {
                                ops.push(HOp::HessEmit {
                                    hess_ptr: ptr as u32,
                                    adj_dot_slot: wd,
                                });
                            }
                        }
                        continue;
                    }
                    TapeOp::Add(a, b) => HOp::RevAdd {
                        adj_a: adj_slot(a as u32),
                        adj_b: adj_slot(b as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        adj_dot_b: adj_dot_slot(b as u32),
                        w,
                        wd,
                    },
                    TapeOp::Sub(a, b) => HOp::RevSub {
                        adj_a: adj_slot(a as u32),
                        adj_b: adj_slot(b as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        adj_dot_b: adj_dot_slot(b as u32),
                        w,
                        wd,
                    },
                    TapeOp::Mul(a, b) => HOp::RevMul {
                        adj_a: adj_slot(a as u32),
                        adj_b: adj_slot(b as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        adj_dot_b: adj_dot_slot(b as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        vb: v_slot(b as u32),
                        dot_a: dot_slot(a as u32),
                        dot_b: dot_slot(b as u32),
                    },
                    TapeOp::Div(a, b) => HOp::RevDiv {
                        adj_a: adj_slot(a as u32),
                        adj_b: adj_slot(b as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        adj_dot_b: adj_dot_slot(b as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        vb: v_slot(b as u32),
                        dot_a: dot_slot(a as u32),
                        dot_b: dot_slot(b as u32),
                    },
                    TapeOp::Pow(a, b) => HOp::RevPow {
                        adj_a: adj_slot(a as u32),
                        adj_b: adj_slot(b as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        adj_dot_b: adj_dot_slot(b as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        vb: v_slot(b as u32),
                        vd: v_slot(i_u),
                        dot_a: dot_slot(a as u32),
                        dot_b: dot_slot(b as u32),
                    },
                    TapeOp::Neg(a) => HOp::RevNeg {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                    },
                    TapeOp::Abs(a) => HOp::RevAbs {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                    },
                    TapeOp::Sqrt(a) => HOp::RevSqrt {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        vd: v_slot(i_u),
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Exp(a) => HOp::RevExp {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        vd: v_slot(i_u),
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Log(a) => HOp::RevLog {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Log10(a) => HOp::RevLog10 {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Sin(a) => HOp::RevSin {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Cos(a) => HOp::RevCos {
                        adj_a: adj_slot(a as u32),
                        adj_dot_a: adj_dot_slot(a as u32),
                        w,
                        wd,
                        va: v_slot(a as u32),
                        dot_a: dot_slot(a as u32),
                    },
                    TapeOp::Funcall { .. } => panic!(
                        "HessianProgram path does not support AMPL external functions; \
                         use the Tape (build_with_externals) path instead."
                    ),
                    TapeOp::Tan(_)
                    | TapeOp::Atan(_)
                    | TapeOp::Acos(_)
                    | TapeOp::Sinh(_)
                    | TapeOp::Cosh(_)
                    | TapeOp::Tanh(_)
                    | TapeOp::Asin(_)
                    | TapeOp::Acosh(_)
                    | TapeOp::Asinh(_)
                    | TapeOp::Atanh(_)
                    | TapeOp::Atan2(_, _)
                    | TapeOp::Cmp(_, _, _)
                    | TapeOp::And(_, _)
                    | TapeOp::Or(_, _)
                    | TapeOp::Not(_)
                    | TapeOp::Select(_, _, _)
                    | TapeOp::Min(_, _)
                    | TapeOp::Max(_, _) => panic!(
                        "HessianProgram path does not yet support tan/atan/acos, the \
                         other transcendental opcodes, atan2, min/max, or \
                         conditional / logical opcodes; use the Tape \
                         (build_with_externals) interpreter path instead."
                    ),
                };
                ops.push(rev_op);
            }
        }

        HessianProgram {
            ops,
            consts,
            n_slots,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots as usize
    }

    pub fn n_ops(&self) -> usize {
        self.ops.len()
    }

    /// Execute the program. `scratch` is overwritten throughout;
    /// it must be at least [`n_slots`] long. `values` is the
    /// shared Hessian-values buffer the caller is accumulating
    /// into (same semantics as
    /// [`Tape::hessian_accumulate`]'s `values`).
    pub fn execute(&self, x: &[f64], weight: f64, scratch: &mut [f64], values: &mut [f64]) {
        debug_assert!(scratch.len() >= self.n_slots as usize);
        if self.ops.is_empty() || weight == 0.0 {
            return;
        }
        let consts = &self.consts[..];
        for &op in &self.ops {
            match op {
                HOp::FwdLoadVar { dst, x_idx } => {
                    scratch[dst as usize] = x[x_idx as usize];
                }
                HOp::FwdLoadConst { dst, c_idx } => {
                    scratch[dst as usize] = consts[c_idx as usize];
                }
                HOp::FwdAdd { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize] + scratch[b as usize];
                }
                HOp::FwdSub { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize] - scratch[b as usize];
                }
                HOp::FwdMul { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize] * scratch[b as usize];
                }
                HOp::FwdDiv { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize] / scratch[b as usize];
                }
                HOp::FwdPow { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize].powf(scratch[b as usize]);
                }
                HOp::FwdNeg { dst, a } => {
                    scratch[dst as usize] = -scratch[a as usize];
                }
                HOp::FwdAbs { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].abs();
                }
                HOp::FwdSqrt { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].sqrt();
                }
                HOp::FwdExp { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].exp();
                }
                HOp::FwdLog { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].ln();
                }
                HOp::FwdLog10 { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].log10();
                }
                HOp::FwdSin { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].sin();
                }
                HOp::FwdCos { dst, a } => {
                    scratch[dst as usize] = scratch[a as usize].cos();
                }

                HOp::SetZero { dst } => {
                    scratch[dst as usize] = 0.0;
                }
                HOp::SetOne { dst } => {
                    scratch[dst as usize] = 1.0;
                }
                HOp::ZeroRange { start, len } => {
                    let s = start as usize;
                    let e = s + len as usize;
                    scratch[s..e].fill(0.0);
                }

                HOp::DotAdd { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize] + scratch[b as usize];
                }
                HOp::DotSub { dst, a, b } => {
                    scratch[dst as usize] = scratch[a as usize] - scratch[b as usize];
                }
                HOp::DotMul {
                    dst,
                    dot_a,
                    vb,
                    va,
                    dot_b,
                } => {
                    scratch[dst as usize] = scratch[dot_a as usize] * scratch[vb as usize]
                        + scratch[va as usize] * scratch[dot_b as usize];
                }
                HOp::DotDiv {
                    dst,
                    dot_a,
                    vb,
                    va,
                    dot_b,
                } => {
                    let v_b = scratch[vb as usize];
                    scratch[dst as usize] = (scratch[dot_a as usize] * v_b
                        - scratch[va as usize] * scratch[dot_b as usize])
                        / (v_b * v_b);
                }
                HOp::DotSqrt { dst, dot_a, vd } => {
                    let svd = scratch[vd as usize];
                    scratch[dst as usize] = if svd > 0.0 {
                        scratch[dot_a as usize] * 0.5 / svd
                    } else {
                        0.0
                    };
                }
                HOp::DotExp { dst, dot_a, vd } => {
                    scratch[dst as usize] = scratch[dot_a as usize] * scratch[vd as usize];
                }
                HOp::DotLog { dst, dot_a, va } => {
                    scratch[dst as usize] = scratch[dot_a as usize] / scratch[va as usize];
                }
                HOp::DotLog10 { dst, dot_a, va } => {
                    scratch[dst as usize] =
                        scratch[dot_a as usize] / (scratch[va as usize] * std::f64::consts::LN_10);
                }
                HOp::DotSin { dst, dot_a, va } => {
                    scratch[dst as usize] = scratch[dot_a as usize] * scratch[va as usize].cos();
                }
                HOp::DotCos { dst, dot_a, va } => {
                    scratch[dst as usize] = -scratch[dot_a as usize] * scratch[va as usize].sin();
                }
                HOp::DotNeg { dst, dot_a } => {
                    scratch[dst as usize] = -scratch[dot_a as usize];
                }
                HOp::DotAbs { dst, dot_a, va } => {
                    scratch[dst as usize] = if scratch[va as usize] >= 0.0 {
                        scratch[dot_a as usize]
                    } else {
                        -scratch[dot_a as usize]
                    };
                }
                HOp::DotPow {
                    dst,
                    va,
                    vb,
                    vd,
                    dot_a,
                    dot_b,
                } => {
                    let u = scratch[va as usize];
                    let r = scratch[vb as usize];
                    let du = scratch[dot_a as usize];
                    let dr = scratch[dot_b as usize];
                    let mut result = 0.0;
                    if r != 0.0 && u != 0.0 {
                        result += r * u.powf(r - 1.0) * du;
                    }
                    if u > 0.0 {
                        result += scratch[vd as usize] * u.ln() * dr;
                    }
                    scratch[dst as usize] = result;
                }

                HOp::RevAdd {
                    adj_a,
                    adj_b,
                    adj_dot_a,
                    adj_dot_b,
                    w,
                    wd,
                } => {
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] += w_v;
                    scratch[adj_b as usize] += w_v;
                    scratch[adj_dot_a as usize] += wd_v;
                    scratch[adj_dot_b as usize] += wd_v;
                }
                HOp::RevSub {
                    adj_a,
                    adj_b,
                    adj_dot_a,
                    adj_dot_b,
                    w,
                    wd,
                } => {
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] += w_v;
                    scratch[adj_b as usize] -= w_v;
                    scratch[adj_dot_a as usize] += wd_v;
                    scratch[adj_dot_b as usize] -= wd_v;
                }
                HOp::RevMul {
                    adj_a,
                    adj_b,
                    adj_dot_a,
                    adj_dot_b,
                    w,
                    wd,
                    va,
                    vb,
                    dot_a,
                    dot_b,
                } => {
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    let va_v = scratch[va as usize];
                    let vb_v = scratch[vb as usize];
                    let da_v = scratch[dot_a as usize];
                    let db_v = scratch[dot_b as usize];
                    scratch[adj_a as usize] += w_v * vb_v;
                    scratch[adj_b as usize] += w_v * va_v;
                    scratch[adj_dot_a as usize] += wd_v * vb_v + w_v * db_v;
                    scratch[adj_dot_b as usize] += wd_v * va_v + w_v * da_v;
                }
                HOp::RevDiv {
                    adj_a,
                    adj_b,
                    adj_dot_a,
                    adj_dot_b,
                    w,
                    wd,
                    va,
                    vb,
                    dot_a,
                    dot_b,
                } => {
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    let va_v = scratch[va as usize];
                    let vb_v = scratch[vb as usize];
                    let vb2 = vb_v * vb_v;
                    let vb3 = vb2 * vb_v;
                    let da_v = scratch[dot_a as usize];
                    let db_v = scratch[dot_b as usize];
                    scratch[adj_a as usize] += w_v / vb_v;
                    scratch[adj_dot_a as usize] += wd_v / vb_v + w_v * (-db_v / vb2);
                    scratch[adj_b as usize] += w_v * (-va_v / vb2);
                    scratch[adj_dot_b as usize] +=
                        wd_v * (-va_v / vb2) + w_v * (-da_v / vb2 + 2.0 * va_v * db_v / vb3);
                }
                HOp::RevPow {
                    adj_a,
                    adj_b,
                    adj_dot_a,
                    adj_dot_b,
                    w,
                    wd,
                    va,
                    vb,
                    vd,
                    dot_a,
                    dot_b,
                } => {
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    let u = scratch[va as usize];
                    let r = scratch[vb as usize];
                    let du = scratch[dot_a as usize];
                    let dr = scratch[dot_b as usize];
                    if r != 0.0 {
                        if u != 0.0 {
                            let p_a = r * u.powf(r - 1.0);
                            scratch[adj_a as usize] += w_v * p_a;
                            let mut dp_a = dr * u.powf(r - 1.0);
                            if u > 0.0 {
                                dp_a += r * u.powf(r - 1.0) * ((r - 1.0) * du / u + dr * u.ln());
                            } else {
                                dp_a += r * (r - 1.0) * u.powf(r - 2.0) * du;
                            }
                            scratch[adj_dot_a as usize] += wd_v * p_a + w_v * dp_a;
                        } else if r >= 2.0 {
                            let p_a = 0.0;
                            scratch[adj_a as usize] += w_v * p_a;
                            let dp_a = if r == 2.0 {
                                2.0 * du
                            } else {
                                r * (r - 1.0) * (0.0_f64).powf(r - 2.0) * du
                            };
                            scratch[adj_dot_a as usize] += wd_v * p_a + w_v * dp_a;
                        }
                    }
                    if u > 0.0 {
                        let ln_u = u.ln();
                        let p_b = scratch[vd as usize] * ln_u;
                        scratch[adj_b as usize] += w_v * p_b;
                        let dur = scratch[vd as usize] * (r * du / u + dr * ln_u);
                        let dp_b = dur * ln_u + scratch[vd as usize] * du / u;
                        scratch[adj_dot_b as usize] += wd_v * p_b + w_v * dp_b;
                    }
                }
                HOp::RevNeg {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                } => {
                    scratch[adj_a as usize] -= scratch[w as usize];
                    scratch[adj_dot_a as usize] -= scratch[wd as usize];
                }
                HOp::RevAbs {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    va,
                } => {
                    let s = if scratch[va as usize] >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    scratch[adj_a as usize] += scratch[w as usize] * s;
                    scratch[adj_dot_a as usize] += scratch[wd as usize] * s;
                }
                HOp::RevSqrt {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    va: _,
                    vd,
                    dot_a,
                } => {
                    let sv = scratch[vd as usize];
                    if sv > 0.0 {
                        let fp = 0.5 / sv;
                        let fpp = -0.25 / (sv * sv * sv);
                        let w_v = scratch[w as usize];
                        let wd_v = scratch[wd as usize];
                        scratch[adj_a as usize] += w_v * fp;
                        scratch[adj_dot_a as usize] +=
                            wd_v * fp + w_v * fpp * scratch[dot_a as usize];
                    }
                }
                HOp::RevExp {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    vd,
                    dot_a,
                } => {
                    let ev = scratch[vd as usize];
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] += w_v * ev;
                    scratch[adj_dot_a as usize] += wd_v * ev + w_v * ev * scratch[dot_a as usize];
                }
                HOp::RevLog {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    va,
                    dot_a,
                } => {
                    let u = scratch[va as usize];
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] += w_v / u;
                    scratch[adj_dot_a as usize] +=
                        wd_v / u + w_v * (-1.0 / (u * u)) * scratch[dot_a as usize];
                }
                HOp::RevLog10 {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    va,
                    dot_a,
                } => {
                    let u = scratch[va as usize];
                    let c = std::f64::consts::LN_10;
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] += w_v / (u * c);
                    scratch[adj_dot_a as usize] +=
                        wd_v / (u * c) + w_v * (-1.0 / (u * u * c)) * scratch[dot_a as usize];
                }
                HOp::RevSin {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    va,
                    dot_a,
                } => {
                    let u = scratch[va as usize];
                    let cu = u.cos();
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] += w_v * cu;
                    scratch[adj_dot_a as usize] +=
                        wd_v * cu + w_v * (-u.sin()) * scratch[dot_a as usize];
                }
                HOp::RevCos {
                    adj_a,
                    adj_dot_a,
                    w,
                    wd,
                    va,
                    dot_a,
                } => {
                    let u = scratch[va as usize];
                    let su = u.sin();
                    let w_v = scratch[w as usize];
                    let wd_v = scratch[wd as usize];
                    scratch[adj_a as usize] -= w_v * su;
                    scratch[adj_dot_a as usize] +=
                        wd_v * (-su) + w_v * (-u.cos()) * scratch[dot_a as usize];
                }

                HOp::HessEmit {
                    hess_ptr,
                    adj_dot_slot,
                } => {
                    values[hess_ptr as usize] += weight * scratch[adj_dot_slot as usize];
                }
            }
        }
    }
}

/// `out[i]` = does tape slot `i` contribute (transitively) to the
/// output slot `n-1`. Used to skip emitting reverse-pass ops for
/// dead slots.
fn reachable_to_output(tape: &Tape) -> Vec<bool> {
    let n = tape.ops.len();
    let mut r = vec![false; n];
    if n == 0 {
        return r;
    }
    r[n - 1] = true;
    for i in (0..n).rev() {
        if !r[i] {
            continue;
        }
        match tape.ops[i] {
            TapeOp::Const(_) | TapeOp::Var(_) => {}
            TapeOp::Add(a, b)
            | TapeOp::Sub(a, b)
            | TapeOp::Mul(a, b)
            | TapeOp::Div(a, b)
            | TapeOp::Pow(a, b)
            | TapeOp::Atan2(a, b) => {
                r[a] = true;
                r[b] = true;
            }
            TapeOp::Neg(a)
            | TapeOp::Abs(a)
            | TapeOp::Sqrt(a)
            | TapeOp::Exp(a)
            | TapeOp::Log(a)
            | TapeOp::Log10(a)
            | TapeOp::Sin(a)
            | TapeOp::Cos(a)
            | TapeOp::Tan(a)
            | TapeOp::Atan(a)
            | TapeOp::Acos(a)
            | TapeOp::Sinh(a)
            | TapeOp::Cosh(a)
            | TapeOp::Tanh(a)
            | TapeOp::Asin(a)
            | TapeOp::Acosh(a)
            | TapeOp::Asinh(a)
            | TapeOp::Atanh(a) => {
                r[a] = true;
            }
            TapeOp::Funcall { .. } => panic!(
                "HessianProgram path does not support AMPL external functions; \
                 use the Tape (build_with_externals) path instead."
            ),
            TapeOp::Cmp(_, _, _)
            | TapeOp::And(_, _)
            | TapeOp::Or(_, _)
            | TapeOp::Not(_)
            | TapeOp::Select(_, _, _)
            | TapeOp::Min(_, _)
            | TapeOp::Max(_, _) => panic!(
                "HessianProgram path does not support conditional / logical / min-max \
                 opcodes; use the Tape (build_with_externals) path instead."
            ),
        }
    }
    r
}

/// `out[i]` = does tape slot `i` transitively read from `Var(j)`.
/// Used to prune forward-tangent ops (slots with `out[i] = false`
/// have `dot[i] = 0` and the rest of the per-`j` pass can skip
/// them).
fn depends_on_var(tape: &Tape, j: usize) -> Vec<bool> {
    let n = tape.ops.len();
    let mut d = vec![false; n];
    for (i, op) in tape.ops.iter().enumerate() {
        d[i] = match *op {
            TapeOp::Const(_) => false,
            TapeOp::Var(k) => k == j,
            TapeOp::Add(a, b)
            | TapeOp::Sub(a, b)
            | TapeOp::Mul(a, b)
            | TapeOp::Div(a, b)
            | TapeOp::Pow(a, b)
            | TapeOp::Atan2(a, b) => d[a] || d[b],
            TapeOp::Neg(a)
            | TapeOp::Abs(a)
            | TapeOp::Sqrt(a)
            | TapeOp::Exp(a)
            | TapeOp::Log(a)
            | TapeOp::Log10(a)
            | TapeOp::Sin(a)
            | TapeOp::Cos(a)
            | TapeOp::Tan(a)
            | TapeOp::Atan(a)
            | TapeOp::Acos(a)
            | TapeOp::Sinh(a)
            | TapeOp::Cosh(a)
            | TapeOp::Tanh(a)
            | TapeOp::Asin(a)
            | TapeOp::Acosh(a)
            | TapeOp::Asinh(a)
            | TapeOp::Atanh(a) => d[a],
            TapeOp::Funcall { .. } => panic!(
                "HessianProgram path does not support AMPL external functions; \
                 use the Tape (build_with_externals) path instead."
            ),
            TapeOp::Cmp(_, _, _)
            | TapeOp::And(_, _)
            | TapeOp::Or(_, _)
            | TapeOp::Not(_)
            | TapeOp::Select(_, _, _)
            | TapeOp::Min(_, _)
            | TapeOp::Max(_, _) => panic!(
                "HessianProgram path does not support conditional / logical / min-max \
                 opcodes; use the Tape (build_with_externals) path instead."
            ),
        };
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, UnaryOp};
    use std::collections::BTreeSet;
    use std::rc::Rc;

    fn cnst(c: f64) -> Expr {
        Expr::Const(c)
    }
    fn var(i: usize) -> Expr {
        Expr::Var(i)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Add, Box::new(a), Box::new(b))
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn pow(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Pow, Box::new(a), Box::new(b))
    }
    fn div(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Div, Box::new(a), Box::new(b))
    }
    fn sub(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b))
    }
    fn unary(op: UnaryOp, a: Expr) -> Expr {
        Expr::Unary(op, Box::new(a))
    }

    /// Build the same shared (row, col) -> pos map both AD paths
    /// scatter into. Lower-triangle pairs, in tape.variables() order.
    fn build_hess_map(tape: &Tape) -> (HashMap<(usize, usize), usize>, Vec<(usize, usize)>) {
        let vars = tape.variables();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut map: HashMap<(usize, usize), usize> = HashMap::new();
        for (ai, &vi) in vars.iter().enumerate() {
            for &vj in &vars[..=ai] {
                let (r, c) = if vi >= vj { (vi, vj) } else { (vj, vi) };
                map.entry((r, c)).or_insert_with(|| {
                    let p = pairs.len();
                    pairs.push((r, c));
                    p
                });
            }
        }
        (map, pairs)
    }

    /// Run both implementations against the same input and assert
    /// values match to a tight ULP-aligned tolerance.
    fn assert_program_matches_tape(tape: &Tape, x: &[f64], weight: f64) {
        let (hess_map, pairs) = build_hess_map(tape);
        let nnz = pairs.len();

        let mut tape_vals = vec![0.0; nnz];
        tape.hessian_accumulate(x, weight, &hess_map, &mut tape_vals);

        let program = HessianProgram::compile(tape, &hess_map);
        let mut scratch = vec![0.0; program.n_slots()];
        let mut prog_vals = vec![0.0; nnz];
        program.execute(x, weight, &mut scratch, &mut prog_vals);

        for (k, &(r, c)) in pairs.iter().enumerate() {
            let tol = tape_vals[k].abs().max(1.0) * 1e-12;
            assert!(
                (tape_vals[k] - prog_vals[k]).abs() < tol,
                "H[{},{}]: tape={:.6e} prog={:.6e}",
                r,
                c,
                tape_vals[k],
                prog_vals[k]
            );
        }
    }

    #[test]
    fn matches_quadratic() {
        let e = add(
            add(
                mul(cnst(3.0), pow(var(0), cnst(2.0))),
                mul(cnst(2.0), mul(var(0), var(1))),
            ),
            pow(var(1), cnst(2.0)),
        );
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[2.0, 3.0], 1.0);
        assert_program_matches_tape(&tape, &[-1.5, 0.7], 2.5);
    }

    #[test]
    fn matches_transcendental() {
        let e = Expr::Sum(vec![
            unary(UnaryOp::Exp, var(0)),
            unary(UnaryOp::Sin, var(1)),
            unary(UnaryOp::Log, var(0)),
            unary(UnaryOp::Sqrt, var(1)),
            mul(var(0), var(1)),
            unary(UnaryOp::Cos, add(var(0), var(1))),
        ]);
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[1.5, 2.0], 1.0);
        assert_program_matches_tape(&tape, &[0.3, 4.1], -0.4);
    }

    #[test]
    fn matches_division() {
        let e = add(div(var(0), var(1)), unary(UnaryOp::Cos, var(0)));
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[0.5, 1.2], 1.0);
    }

    #[test]
    fn matches_through_cse() {
        let body = Rc::new(add(var(0), var(1)));
        let e = add(
            pow(Expr::Cse(body.clone()), cnst(2.0)),
            Expr::Cse(body.clone()),
        );
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[1.0, 2.0], 1.0);
        assert_program_matches_tape(&tape, &[-0.5, 3.3], 0.7);
    }

    #[test]
    fn matches_pow_chain() {
        // After Tier 1 this lowers to a Mul chain; verify both
        // paths agree on the lowered form too.
        let e = add(pow(var(0), cnst(3.0)), pow(var(1), cnst(-2.0)));
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[1.7, 0.8], 1.0);
    }

    #[test]
    fn matches_residual_pow_with_var_exponent() {
        // Pow where the exponent is variable (not constant), so
        // it survives Tier 1 and exercises the RevPow / DotPow
        // compound branches.
        let e = pow(var(0), var(1));
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[2.5, 1.4], 1.0);
        assert_program_matches_tape(&tape, &[0.6, 2.1], -1.0);
    }

    #[test]
    fn matches_sub_neg_abs() {
        let e = sub(
            unary(UnaryOp::Neg, var(0)),
            unary(UnaryOp::Abs, sub(var(1), var(0))),
        );
        let tape = Tape::build(&e);
        assert_program_matches_tape(&tape, &[1.0, -2.0], 1.0);
        assert_program_matches_tape(&tape, &[-3.5, 4.0], 0.9);
    }

    #[test]
    fn slots_layout_matches_design() {
        let e = mul(var(0), var(1));
        let tape = Tape::build(&e);
        let (hess_map, _) = build_hess_map(&tape);
        let prog = HessianProgram::compile(&tape, &hess_map);
        assert_eq!(prog.n_slots(), 4 * tape.ops.len());
    }

    /// Sanity: the pruning analyses are consistent with the slot
    /// structure exposed via `hessian_sparsity()`.
    #[test]
    fn dependence_matches_hessian_sparsity_for_simple_case() {
        let e = add(unary(UnaryOp::Sin, var(0)), mul(var(1), var(2)));
        let tape = Tape::build(&e);
        let s: BTreeSet<(usize, usize)> = tape.hessian_sparsity();
        // (0,0) from sin, (2,1) from x1*x2, (1,1)/(2,2) NOT there
        // because Mul(x1, x2) emits cross only.
        assert!(s.contains(&(0, 0)));
        assert!(s.contains(&(2, 1)));
        assert_program_matches_tape(&tape, &[0.7, 1.1, 2.2], 1.0);
    }
}
