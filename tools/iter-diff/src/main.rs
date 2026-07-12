//! `iter-diff` — compare two POUNCEIT iterate-trace files.
//!
//! Usage: see `--help`. Exit code 0 on match, 1 on divergence, 2 on
//! parse / I/O error.

use iter_diff::{CompareError, Tolerance, compare, parse_file};
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    left: PathBuf,
    right: PathBuf,
    tolerance: Tolerance,
    show_first_divergence: bool,
    strict: bool,
}

fn print_help() {
    println!(
        "iter-diff — compare two POUNCEIT iterate-trace files\n\
         \n\
         Usage:\n  \
           iter-diff --left <path> --right <path> [options]\n\
         \n\
         Options:\n  \
           --tolerance <bit|ulp:N|abs:F|rel:F>   Comparison mode (default: bit)\n  \
           --show-first-divergence               Stop at the first mismatch\n  \
           --strict                              Compare advisory fields\n  \
                                                 (delta_s/c/d, filter, nnz)\n  \
           -h, --help                            Show this help\n\
         \n\
         Exit codes:\n  \
           0 — match within tolerance\n  \
           1 — divergence detected\n  \
           2 — parse / I/O error or bad arguments"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut left: Option<PathBuf> = None;
    let mut right: Option<PathBuf> = None;
    let mut tolerance = Tolerance::Bit;
    let mut show_first_divergence = false;
    let mut strict = false;
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--left" => {
                let v = argv.next().ok_or("--left requires a value")?;
                left = Some(PathBuf::from(v));
            }
            "--right" => {
                let v = argv.next().ok_or("--right requires a value")?;
                right = Some(PathBuf::from(v));
            }
            "--tolerance" => {
                let v = argv.next().ok_or("--tolerance requires a value")?;
                tolerance = Tolerance::parse(&v)?;
            }
            "--show-first-divergence" => show_first_divergence = true,
            "--strict" => strict = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{}`", other)),
        }
    }
    let left = left.ok_or("missing --left")?;
    let right = right.ok_or("missing --right")?;
    Ok(Args {
        left,
        right,
        tolerance,
        show_first_divergence,
        strict,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("iter-diff: {}", e);
            eprintln!("run `iter-diff --help` for usage");
            return ExitCode::from(2);
        }
    };

    let left = match parse_file(&args.left) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("iter-diff: parse {}: {}", args.left.display(), e);
            return ExitCode::from(2);
        }
    };
    let right = match parse_file(&args.right) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("iter-diff: parse {}: {}", args.right.display(), e);
            return ExitCode::from(2);
        }
    };

    let divs = compare(
        &left,
        &right,
        args.tolerance,
        args.strict,
        args.show_first_divergence,
    );
    if divs.is_empty() {
        println!(
            "match: left={} right={} ({} record(s), tolerance={:?})",
            args.left.display(),
            args.right.display(),
            left.1.len(),
            args.tolerance
        );
        return ExitCode::from(0);
    }
    eprintln!(
        "DIVERGENCE: left={} right={} ({} issue(s))",
        args.left.display(),
        args.right.display(),
        divs.len()
    );
    for d in &divs {
        eprintln!("  {}", d);
        if matches!(
            d,
            CompareError::Header(_) | CompareError::RecordCount { .. }
        ) {
            // header/count issues are most actionable; still print the rest
        }
    }
    ExitCode::from(1)
}
