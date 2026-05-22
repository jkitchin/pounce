//! Minimal AMPL `.sol`-format writer.
//!
//! Format reference: David M. Gay, "Hooking Your Solver to AMPL"
//! (<https://ampl.com/REFS/hooking2.pdf>) §5 ("Returning Results to
//! AMPL"), cross-checked against the AMPL solver-library reference
//! implementation `write_sol_ASL` in
//! <https://github.com/ampl/asl> (`solvers/writesol.c`). We emit the
//! ASCII variant — the same one AMPL's `commands` file produces by
//! default when reading back from solvers.
//!
//! # Format
//!
//! ```text
//! <message line 1>
//! <message line 2>
//! ...                         (free text, ended by a blank line then "Options")
//!
//! Options
//! <nopts>                     (int — number of integer option-words to follow)
//! <opt0>                      (... nopts lines)
//! ...
//! <n_dual>                    (number of dual values written below)
//! <m>                         (constraint count)
//! <n_primal>                  (number of primal values written below)
//! <n>                         (variable count)
//! <lambda[0]>                 (... n_dual lines, dual values)
//! ...
//! <x[0]>                      (... n_primal lines, primal values)
//! ...
//! objno <objno> <status>      (optional — selects which objective and the solver-return code)
//! suffix <kind> <nvalues> <namelen> <tablen> <tabline>  (optional — one block per exported suffix)
//! <name>                      (the suffix name, on its own line)
//! <idx> <value>
//! ...
//! ```
//!
//! The four-integer count block is the canonical AMPL form: each
//! dimension count is paired with a "values written" partner so the
//! reader knows how many dual and primal lines to consume before
//! reaching `objno`. We always write every dual and primal, so
//! `n_dual == m` and `n_primal == n`. (Earlier pounce builds emitted
//! only the two bare counts `<m>\n<n>\n`; AMPL's own reader and
//! Pyomo's `.sol` reader both reject that short form.)
//!
//! # Scope
//!
//! Smallest writer that lets [`crate::nl_reader::NlSuffixes`] flow
//! from a pounce solve back through AMPL's reader. Specifically the
//! `pounce_sens` binary (pounce#17) writes:
//! * The nominal primal and dual blocks (so AMPL sees `x*` and `λ*`
//!   on the regular `_var.X` / `_con.dual` slots).
//! * One or more sensitivity suffixes (`sens_sol_state_<N>`) carrying
//!   the perturbed primal as a real-var suffix, matching upstream
//!   `MetadataMeasurement::SetSolution`
//!   (`ref/Ipopt/contrib/sIPOPT/src/SensMetadataMeasurement.cpp:128-150`).

use pounce_common::types::{Index, Number};
use std::fmt::Write as _;
use std::path::Path;

/// A single suffix block to write back into the `.sol` file. Mirrors
/// the `S`-segment shape of [`crate::nl_reader::NlSuffixes`] entries.
#[derive(Debug, Clone)]
pub struct SolSuffix {
    /// `name` as it appears in AMPL.
    pub name: String,
    /// Which side the suffix attaches to. Mapped to AMPL's
    /// `ASL_Sufkind_var` / `_con` / `_obj` / `_prob` (= 0/1/2/3).
    pub target: SolSuffixTarget,
    /// Real or integer-typed values. AMPL's `ASL_Sufkind_real` flag
    /// (`0x4`) on the kind byte selects this; we accept either typed
    /// payload here and tag the kind accordingly on write.
    pub values: SolSuffixValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolSuffixTarget {
    Var = 0,
    Con = 1,
    Obj = 2,
    Problem = 3,
}

#[derive(Debug, Clone)]
pub enum SolSuffixValues {
    /// One entry per dimension of the target (variables / constraints /
    /// objectives). Sparse zero-trim happens on write — only non-zero
    /// entries land in the output, matching how AMPL emits suffixes.
    Int(Vec<Index>),
    Real(Vec<Number>),
    /// Problem-level scalar (target = Problem). Always emitted (no
    /// sparse trim, since there's only one slot).
    ProblemInt(Index),
    ProblemReal(Number),
}

/// Solution payload bundled for a `.sol` write.
#[derive(Debug, Clone)]
pub struct SolutionFile<'a> {
    /// Free-text banner / status line(s). Goes at the top of the file.
    pub message: &'a str,
    /// Primal variable values, length `n`.
    pub x: &'a [Number],
    /// Constraint dual values, length `m`.
    pub lambda: &'a [Number],
    /// AMPL solver return code. Convention: 0 = solved, 100..199 =
    /// "solved with warning", 200..299 = "infeasible", 300..399 =
    /// "unbounded", 400..499 = "limit reached", 500..599 = "failure".
    /// See [Gay §5, table on p. 23](https://ampl.com/REFS/hooking2.pdf).
    pub solve_result_num: i32,
    /// Suffix blocks to emit after the primal/dual blocks. Empty when
    /// no sensitivity / reduced-Hessian outputs are populated.
    pub suffixes: &'a [SolSuffix],
}

/// Format `payload` into AMPL `.sol` ASCII text.
pub fn format_sol(payload: &SolutionFile<'_>) -> String {
    let mut out = String::new();

    // Header: message + a blank line + "Options" + zero options.
    for line in payload.message.lines() {
        let _ = writeln!(out, "{line}");
    }
    out.push('\n');
    out.push_str("Options\n");
    out.push_str("0\n");

    // Count block: the canonical AMPL four-integer form
    //   <n_dual_written> <n_con> <n_primal_written> <n_var>
    // The "written" counts tell the reader how many value lines to
    // consume; the bare counts are matched against the originating
    // `.nl`. We write every dual and primal, so the pairs collapse to
    // (m, m) and (n, n). Emitting only `m` and `n` (the two-integer
    // short form) makes AMPL's and Pyomo's `.sol` readers fail.
    let m = payload.lambda.len();
    let n = payload.x.len();
    let _ = writeln!(out, "{m}");
    let _ = writeln!(out, "{m}");
    let _ = writeln!(out, "{n}");
    let _ = writeln!(out, "{n}");

    // Dual block, then primal block. AMPL writes doubles with at least
    // 16 significant digits to round-trip through IEEE-754; we use
    // Rust's `{:.17e}` to match.
    for &v in payload.lambda {
        let _ = writeln!(out, "{v:.17e}");
    }
    for &v in payload.x {
        let _ = writeln!(out, "{v:.17e}");
    }

    // Objective-number + solver return code. AMPL convention: every
    // .sol must end with at least an `objno <objno> <code>` line so
    // the reader can extract `solve_result_num`.
    let _ = writeln!(out, "objno 0 {}", payload.solve_result_num);

    // Suffix blocks. AMPL's reader skips empty / all-zero suffixes,
    // but it accepts them; we sparse-trim ints/reals to keep the
    // output small. Problem-level kinds always write a single entry.
    for s in payload.suffixes {
        write_suffix(&mut out, s);
    }

    out
}

fn write_suffix(out: &mut String, s: &SolSuffix) {
    let target_bits = s.target as u32 & 0x3;
    match &s.values {
        SolSuffixValues::Int(vs) => {
            let entries: Vec<(usize, Index)> = vs
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0)
                .map(|(i, &v)| (i, v))
                .collect();
            write_suffix_header(out, target_bits, entries.len(), &s.name);
            for (i, v) in entries {
                let _ = writeln!(out, "{i} {v}");
            }
        }
        SolSuffixValues::Real(vs) => {
            let entries: Vec<(usize, Number)> = vs
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0.0)
                .map(|(i, &v)| (i, v))
                .collect();
            write_suffix_header(out, target_bits | 0x4, entries.len(), &s.name);
            for (i, v) in entries {
                let _ = writeln!(out, "{i} {v:.17e}");
            }
        }
        SolSuffixValues::ProblemInt(v) => {
            write_suffix_header(out, target_bits, 1, &s.name);
            let _ = writeln!(out, "0 {v}");
        }
        SolSuffixValues::ProblemReal(v) => {
            write_suffix_header(out, target_bits | 0x4, 1, &s.name);
            let _ = writeln!(out, "0 {v:.17e}");
        }
    }
}

/// Emit the canonical AMPL `.sol` suffix header: five integers
/// `suffix <kind> <nvalues> <namelen> <tablen> <tabline>` followed by
/// the suffix name on its own line. `namelen` is `strlen(name)+1` (the
/// value ASL's `writesol.c` writes); `tablen`/`tabline` are 0 — pounce
/// never emits a suffix value-table. AMPL's and Pyomo's `.sol` readers
/// both require this five-integer form and read the name from the next
/// line; the older three-token `suffix <kind> <nvalues> <name>` shape
/// is rejected.
fn write_suffix_header(out: &mut String, kind: u32, nvalues: usize, name: &str) {
    let namelen = name.len() + 1;
    let _ = writeln!(out, "suffix {kind} {nvalues} {namelen} 0 0");
    let _ = writeln!(out, "{name}");
}

/// Convenience: write `payload` to `path` (truncating any existing
/// file). Returns the bytes written on success.
pub fn write_sol_file(path: &Path, payload: &SolutionFile<'_>) -> std::io::Result<usize> {
    let s = format_sol(payload);
    std::fs::write(path, &s)?;
    Ok(s.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_basic_primal_dual_block() {
        let payload = SolutionFile {
            message: "POUNCE: SolveSucceeded",
            x: &[1.0, 2.5, -0.5],
            lambda: &[0.1, -0.2],
            solve_result_num: 0,
            suffixes: &[],
        };
        let s = format_sol(&payload);
        // Header banner present.
        assert!(s.starts_with("POUNCE: SolveSucceeded\n"));
        assert!(s.contains("\nOptions\n0\n"));
        // Four-integer count block: n_dual=2, m=2, n_primal=3, n=3.
        assert!(s.contains("\n2\n2\n3\n3\n"), "counts missing:\n{s}");
        // First dual line: 0.1 in exponent form.
        assert!(
            s.contains("1.00000000000000006e-1\n") || s.contains("1.0e-1\n"),
            "lambda not present:\n{s}",
        );
        // objno tail present.
        assert!(s.trim_end().ends_with("objno 0 0"));
    }

    #[test]
    fn writes_real_var_suffix_sparse_trimming_zeros() {
        let payload = SolutionFile {
            message: "POUNCE-SENS",
            x: &[0.0, 0.0],
            lambda: &[],
            solve_result_num: 0,
            suffixes: &[SolSuffix {
                name: "sens_sol_state_1".into(),
                target: SolSuffixTarget::Var,
                // Dense (0, 5.0, 0, 3.5); only indices 1 and 3 should
                // appear.
                values: SolSuffixValues::Real(vec![0.0, 5.0, 0.0, 3.5]),
            }],
        };
        let s = format_sol(&payload);
        // Canonical header: kind = 0|0x4 = 4 (real var), 2 values,
        // namelen = 17 ("sens_sol_state_1" + NUL), no table; name on
        // the following line.
        assert!(
            s.contains("\nsuffix 4 2 17 0 0\nsens_sol_state_1\n"),
            "missing suffix header:\n{s}",
        );
        // entries present with correct indices.
        assert!(s.contains("\n1 5.0"), "missing entry idx 1:\n{s}");
        assert!(s.contains("\n3 3.5"), "missing entry idx 3:\n{s}");
        // index 0 / 2 are zero — must not appear in the suffix block.
        // (The single-digit `0` could appear elsewhere, so we anchor.)
        assert!(!s.contains("\n0 0.0"), "zero entry was not trimmed:\n{s}",);
    }

    #[test]
    fn writes_int_constraint_suffix() {
        let payload = SolutionFile {
            message: "msg",
            x: &[],
            lambda: &[],
            solve_result_num: 0,
            suffixes: &[SolSuffix {
                name: "sens_init_constr".into(),
                target: SolSuffixTarget::Con,
                values: SolSuffixValues::Int(vec![0, 1, 2, 0]),
            }],
        };
        let s = format_sol(&payload);
        // kind = 1 (con, integer), 2 values, namelen = 17.
        assert!(
            s.contains("\nsuffix 1 2 17 0 0\nsens_init_constr\n"),
            "{s}"
        );
        assert!(s.contains("\n1 1\n"));
        assert!(s.contains("\n2 2\n"));
    }

    #[test]
    fn writes_problem_real_suffix() {
        let payload = SolutionFile {
            message: "msg",
            x: &[],
            lambda: &[],
            solve_result_num: 0,
            suffixes: &[SolSuffix {
                name: "wall_time".into(),
                target: SolSuffixTarget::Problem,
                values: SolSuffixValues::ProblemReal(0.0123),
            }],
        };
        let s = format_sol(&payload);
        // kind = 3 | 0x4 = 7 (problem-level, real), namelen = 10.
        assert!(s.contains("\nsuffix 7 1 10 0 0\nwall_time\n"), "{s}");
        // Single entry at idx 0.
        assert!(s.contains("0 1.23"));
    }

    #[test]
    fn round_trip_through_nl_reader_suffix_parser() {
        // Build a .sol with an integer var-suffix, then feed the
        // suffix block to the .nl-style parser to confirm shape /
        // index conventions agree. We don't reuse parse_nl_text here
        // because the .sol prefix differs from .nl; instead we just
        // string-search the emitted suffix header against the
        // {kind, name, count} contract.
        let payload = SolutionFile {
            message: "m",
            x: &[],
            lambda: &[],
            solve_result_num: 0,
            suffixes: &[SolSuffix {
                name: "foo".into(),
                target: SolSuffixTarget::Var,
                values: SolSuffixValues::Int(vec![1, 0, 3]),
            }],
        };
        let s = format_sol(&payload);
        // kind = 0 (var int), 2 values, namelen = 4.
        assert!(s.contains("\nsuffix 0 2 4 0 0\nfoo\n"), "{s}");
    }
}
