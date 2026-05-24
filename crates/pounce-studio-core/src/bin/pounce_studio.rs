//! `pounce-studio` — Markdown inspector for solve reports and iter-dumps.
//!
//! ```text
//! pounce-studio inspect <report.json>      # JSON solve report → Markdown
//! pounce-studio dump-summary <trace.bin>   # POUNCEIT v1 binary → summary
//! pounce-studio version
//! ```
//!
//! Reads from disk, prints to stdout. The library that does all the
//! actual work (`pounce-studio-core`) is WASM-clean — see `src/lib.rs`.

use std::path::Path;
use std::process::ExitCode;

use pounce_studio_core::{iter_dump::IterDumpTrace, markdown::render_inspect, report::SolveReport};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("inspect") => match args.get(1) {
            Some(path) => match cmd_inspect(Path::new(path)) {
                Ok(md) => {
                    print!("{md}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("pounce-studio: {e}");
                    ExitCode::from(2)
                }
            },
            None => {
                eprintln!("pounce-studio inspect: missing path argument");
                ExitCode::from(2)
            }
        },
        Some("dump-summary") => match args.get(1) {
            Some(path) => match cmd_dump_summary(Path::new(path)) {
                Ok(md) => {
                    print!("{md}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("pounce-studio: {e}");
                    ExitCode::from(2)
                }
            },
            None => {
                eprintln!("pounce-studio dump-summary: missing path argument");
                ExitCode::from(2)
            }
        },
        Some("version") | Some("--version") | Some("-V") => {
            println!("pounce-studio {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("help") | Some("--help") | Some("-h") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("pounce-studio: unknown subcommand {other:?}");
            print_help();
            ExitCode::from(2)
        }
    }
}

fn cmd_inspect(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let report = SolveReport::from_json_slice(&bytes)?;
    Ok(render_inspect(&report))
}

fn cmd_dump_summary(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let trace = IterDumpTrace::from_bytes(&bytes)?;
    let mut out = String::new();
    use std::fmt::Write as _;
    writeln!(out, "# POUNCEIT v{} trace", trace.header.format_version)?;
    writeln!(out)?;
    // header.name comes from the writer-side env var `IPOPT_ITER_DUMP_NAME`
    // and could legitimately carry odd characters — sanitise before
    // dropping it into an inline code span.
    writeln!(
        out,
        "- **name**: `{}`",
        trace.header.name.replace('`', "\u{02CB}"),
    )?;
    writeln!(
        out,
        "- **n** (variables): {}, **m** (constraints): {}",
        trace.header.n, trace.header.m,
    )?;
    writeln!(out, "- **records**: {}", trace.records.len())?;
    writeln!(out)?;
    writeln!(out, "| iter | mu | inf_pr | inf_du | α_pr | α_du | f |")?;
    writeln!(out, "|---|---|---|---|---|---|---|")?;
    // Cap printed rows for readability; show first 20 and last 5 with
    // an elision row in between when the trace is long. Matches the
    // policy used by the JSON-side `inspect` Markdown renderer.
    let n = trace.records.len();
    let to_show: Vec<usize> = if n <= 30 {
        (0..n).collect()
    } else {
        (0..20).chain((n - 5)..n).collect()
    };
    let mut last: Option<usize> = None;
    for i in to_show {
        if let Some(prev) = last {
            if i != prev + 1 {
                writeln!(out, "| ... | | | | | | |")?;
            }
        }
        let r = &trace.records[i];
        writeln!(
            out,
            "| {} | {:.2e} | {:.2e} | {:.2e} | {:.3} | {:.3} | {:.6e} |",
            r.iter, r.mu, r.inf_pr, r.inf_du, r.alpha_pr, r.alpha_du, r.f,
        )?;
        last = Some(i);
    }
    Ok(out)
}

fn print_help() {
    println!(
        "pounce-studio — inspector for pounce solve reports and iter-dumps\n\
\n\
USAGE:\n  \
  pounce-studio <COMMAND> [ARGS]\n\
\n\
COMMANDS:\n  \
  inspect <report.json>       Render a Markdown summary of a JSON solve report\n  \
  dump-summary <trace.bin>    Render a Markdown summary of a POUNCEIT v1 trace\n  \
  version                     Print the crate version\n  \
  help                        Show this message\n"
    );
}
