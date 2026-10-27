//! gh#819 — the iteration count a restoration-terminating solve reports.
//!
//! Ipopt's summary line `Number of Iterations....:` is the **index of the
//! last printed iteration row**, `r` rows included, on every exit path.
//! Measured, not assumed, over four upstream logs from this repo's issue
//! corpus: `2418r`/2418 (Restoration Failed), `412r`/412 (local
//! infeasibility), `1547r`/1547 and `1348r`/1348.
//!
//! POUNCE printed the same `r` rows and then reported a number that ignored
//! them. Three separate defects stacked up:
//!
//! 1. The outer `iter_count` roll-forward lived only on the `Recovered` path
//!    (`min_c_1nrm.rs` step 2g). All four *terminating* restoration outcomes
//!    return ahead of it, so a solve that ended inside restoration reported
//!    the outer count at the moment restoration was entered — the whole
//!    sub-solve simply vanished from the summary.
//! 2. `restoration_inner_iters` summed the inner IPM's **absolute**
//!    terminating `iter_count`. That counter is seeded from the outer's
//!    (`inner.iter_count = outer_iter + 1`, mirroring `IpRestoMinC_1Nrm.cpp`
//!    line 181), so the sum was a position in the shared `r`-row numbering,
//!    not a length. gh#664 documents the same misreading for the stall gate.
//! 3. The count was read off the `Some(result)` arm, so on every path that
//!    bailed — which is every path that *ends* in restoration — it recorded
//!    `0`.
//!
//! # What this file pins
//!
//! Both halves of the fix, on the failing path, against a real solve:
//! the reported total equals the last row's index, and the new restoration
//! line equals the number of `r` rows actually printed. Both are parsed out
//! of the log rather than hardcoded, so the assertions survive a trajectory
//! change that moves the numbers without breaking the identity.
//!
//! # What it is *not* evidence about
//!
//! One fixture, one exit path (`Restoration_Failed`), one linear solver.
//!
//! The witness was `tol=1e-11` on the exact leg until gh#1084 showed that
//! exit was a square model at a feasible point below its round-off floor, now
//! reported as `Feasible_Point_Found` (see `issue_815_restoration_ladder.rs`).
//! It is `max_resto_iter=5` now: restoration is entered far from feasible and
//! cannot get there in five iterations, so the run ends `Restoration Failed!`
//! on every platform. The assertions parse the log rather than pin numbers.
//!
//! The
//! `Recovered` path had a working roll-forward before this change and still
//! does; `restoration_deadline.rs` is what covers the multi-call arm. The
//! `Infeasible_Problem_Detected` and `MaximumIterationsExceeded` restoration
//! exits are not reached here.

use std::path::PathBuf;
use std::process::Command;

const MODEL: &str = "square_flowsheet_resto.nl";

/// `max_resto_iter=5` is what puts the fixture on the terminating path
/// (module comment). The ladder is switched off so that the run's last
/// printed rows are the base solve's own: every rung a `Restoration_Failed`
/// can reach is named — the gh#815 displacement rung, gh#857's
/// `feral_increase_quality_retry`, and the square-system stage. The square-system
/// stage recovers this model, and each would print its own iteration
/// table after the base solve's, and the identity is about one solve. A rung
/// added later that catches this trigger belongs here too.
fn run_to_restoration_failure() -> String {
    let mut model = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model.push("tests");
    model.push("fixtures");
    model.push(MODEL);
    let sol = std::env::temp_dir().join("pounce_i819.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&model)
        .arg(&sol)
        .arg("max_resto_iter=5")
        .arg("infeasibility_perturbed_start_retry=no")
        .arg("feral_increase_quality_retry=no")
        .arg("square_system_recovery=no")
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    )
}

/// The leading token of an iteration row: an integer, optionally suffixed
/// `r` for a restoration row. Returns `(index, is_restoration_row)`.
fn iteration_row(line: &str) -> Option<(i64, bool)> {
    let tok = line.split_whitespace().next()?;
    // A row must carry the columns behind the index; the header and the
    // banner lines do not.
    if line.split_whitespace().count() < 5 {
        return None;
    }
    let (digits, is_r) = match tok.strip_suffix('r') {
        Some(d) => (d, true),
        None => (tok, false),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((digits.parse().ok()?, is_r))
}

fn parse_summary_number(log: &str, prefix: &str) -> i64 {
    let line = log
        .lines()
        .find(|l| l.trim_start().starts_with(prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in the summary:\n{log}"));
    let after = line
        .split(&['.', '=', ':'][..])
        .next_back()
        .unwrap_or_default();
    after
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .parse()
        .unwrap_or_else(|_| panic!("cannot read a number out of `{line}`"))
}

#[test]
fn the_reported_count_is_the_last_printed_row_including_restoration_rows() {
    let log = run_to_restoration_failure();
    assert!(
        log.contains("EXIT: Restoration Failed!"),
        "this file is about the terminating-restoration path — if the \
         fixture no longer reaches it every assertion below is vacuous:\n{log}"
    );

    let rows: Vec<(i64, bool)> = log.lines().filter_map(iteration_row).collect();
    let (last_index, last_is_r) = *rows
        .last()
        .unwrap_or_else(|| panic!("no iteration rows were printed at all:\n{log}"));
    assert!(
        last_is_r,
        "a solve that exits `Restoration Failed!` must have a restoration \
         row last; got index {last_index}:\n{log}"
    );

    let reported = parse_summary_number(&log, "Number of Iterations");
    assert_eq!(
        reported, last_index,
        "gh#819: the summary must report the index of the last printed row, \
         `r` rows included — that is Ipopt's rule on every exit path. Before \
         the fix this read the outer count at the moment restoration was \
         entered, which was {reported} against a last row of \
         {last_index}:\n{log}"
    );
}

#[test]
fn the_restoration_line_counts_the_r_rows_that_were_printed() {
    let log = run_to_restoration_failure();
    let rows: Vec<(i64, bool)> = log.lines().filter_map(iteration_row).collect();
    let r_rows = rows.iter().filter(|(_, is_r)| *is_r).count() as i64;
    assert!(
        r_rows > 0,
        "the fixture is supposed to spend real iterations in restoration:\n{log}"
    );

    let reported = parse_summary_number(&log, "Number of restoration iterations");
    assert_eq!(
        reported, r_rows,
        "gh#819: `restoration_inner_iters` is a sub-solve *length*, so the \
         summary line must equal the number of `r` rows printed. Before the \
         fix this was `0` on every terminating path — the count was read off \
         the `Some(result)` arm, which those paths never take — and on the \
         recovering path it summed the inner solver's absolute counter, which \
         is seeded from the outer's and is therefore a position, not a \
         length:\n{log}"
    );
    // The call count is printed with the noun agreeing with it, whatever
    // the count is (one call under this witness).
    let line = log
        .lines()
        .find(|l| l.starts_with("Number of restoration iterations"))
        .expect("restoration summary line");
    let calls: i64 = line
        .split("(in ")
        .nth(1)
        .and_then(|t| t.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no call count in `{line}`"));
    assert!(calls >= 1, "at least one restoration call:\n{log}");
    let noun = if calls == 1 { "call)" } else { "calls)" };
    assert!(
        line.contains(&format!("(in {calls} {noun}")),
        "the call count and its noun must agree: `{line}`"
    );
}

/// The two numbers are not independent: everything before the first `r` row
/// is an ordinary iteration, so the restoration length must be strictly less
/// than the total. A fix that simply set one from the other would pass the
/// two tests above and fail this one.
#[test]
fn the_restoration_span_sits_inside_the_total() {
    let log = run_to_restoration_failure();
    let rows: Vec<(i64, bool)> = log.lines().filter_map(iteration_row).collect();
    let first_r = rows
        .iter()
        .position(|(_, is_r)| *is_r)
        .unwrap_or_else(|| panic!("no restoration rows:\n{log}"));
    let plain_rows = first_r as i64;
    let total = parse_summary_number(&log, "Number of Iterations");
    let resto = parse_summary_number(&log, "Number of restoration iterations");

    assert!(
        resto < total,
        "restoration cannot account for every iteration — the solve reaches \
         it from somewhere ({resto} of {total}):\n{log}"
    );
    assert!(
        plain_rows > 0,
        "the fixture must take ordinary iterations before restoration \
         opens:\n{log}"
    );
}
