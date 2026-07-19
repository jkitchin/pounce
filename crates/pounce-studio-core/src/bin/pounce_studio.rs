//! `pounce-studio` — multi-subcommand inspector for pounce solve
//! reports, iter-dumps, and pre-flight problem files.
//!
//! Most analysis subcommands emit JSON (suitable for piping into `jq`
//! or consuming from a Claude skill). The two human-facing
//! Markdown-renderers — `inspect` and `dump-summary` — keep their
//! Markdown-on-stdout behaviour for shell users; everything else
//! defaults to JSON.
//!
//! Run `pounce-studio help` for the full subcommand list, or
//! `pounce-studio <cmd> --help` for per-command flags.
//!
//! The library that does the actual work (`pounce-studio-core`) is
//! WASM-clean — this binary owns the filesystem touches.

use std::path::Path;
use std::process::ExitCode;

use pounce_studio_core::{
    analysis, glossary, iter_dump::IterDumpTrace, markdown::render_inspect, preflight,
    report::SolveReport,
};

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = raw.first() else {
        print_help();
        return ExitCode::SUCCESS;
    };
    let rest: Vec<String> = raw[1..].to_vec();

    let result: Result<(), CliError> = match cmd.as_str() {
        "inspect" => cmd_inspect(&rest),
        "dump-summary" => cmd_dump_summary(&rest),
        "summary" => cmd_summary(&rest),
        "diagnose" => cmd_diagnose(&rest),
        "find-stalls" => cmd_find_stalls(&rest),
        "convergence-trace" => cmd_convergence_trace(&rest),
        "get-iterate" => cmd_get_iterate(&rest),
        "restoration-windows" => cmd_restoration_windows(&rest),
        "compare" => cmd_compare(&rest),
        "linear-solver-summary" => cmd_linear_solver_summary(&rest),
        "explain" => cmd_explain(&rest),
        "citations" => cmd_citations(&rest),
        "analyze-nl" => cmd_analyze_nl(&rest),
        "analyze-gms" => cmd_analyze_gms(&rest),
        "parse-gams-listing" => cmd_parse_gams_listing(&rest),
        "list-gams" => cmd_list_gams(&rest),
        "list-builtins" => cmd_list_builtins(&rest),
        "version" | "--version" | "-V" => {
            println!("pounce-studio {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(CliError::UnknownCommand(other.into())),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("pounce-studio: {e}");
            if matches!(e, CliError::UnknownCommand(_) | CliError::Usage(_)) {
                eprintln!();
                print_help_to_stderr();
            }
            ExitCode::from(2)
        }
    }
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    UnknownCommand(String),
    Io(std::io::Error),
    Parse(String),
    Domain(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(m) => write!(f, "{m}"),
            Self::UnknownCommand(c) => write!(f, "unknown subcommand {c:?}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Parse(m) => write!(f, "{m}"),
            Self::Domain(m) => write!(f, "{m}"),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---- option-parsing helpers -----------------------------------------

/// Pop a `--name VALUE` pair from `args`. Returns the value when
/// matched and removes both tokens; returns `None` when the flag is
/// absent. Errors when the flag is present but missing its value.
fn take_flag(args: &mut Vec<String>, name: &str) -> Result<Option<String>, CliError> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if i + 1 >= args.len() {
                return Err(CliError::Usage(format!("{name} requires a value")));
            }
            let val = args.remove(i + 1);
            args.remove(i);
            return Ok(Some(val));
        }
        i += 1;
    }
    Ok(None)
}

fn take_bool_flag(args: &mut Vec<String>, name: &str) -> bool {
    if let Some(pos) = args.iter().position(|a| a == name) {
        args.remove(pos);
        true
    } else {
        false
    }
}

fn positionals(args: Vec<String>) -> Vec<String> {
    args
}

fn load_report(path: &str) -> Result<SolveReport, CliError> {
    let bytes = std::fs::read(path).map_err(|e| {
        CliError::Io(std::io::Error::new(
            e.kind(),
            format!("could not read {path:?}: {e}"),
        ))
    })?;
    SolveReport::from_json_slice(&bytes).map_err(|e| CliError::Parse(e.to_string()))
}

/// Pretty-print a serializable value as JSON.
fn emit_json<T: serde::Serialize>(value: &T) -> Result<(), CliError> {
    let s = serde_json::to_string_pretty(value)
        .map_err(|e| CliError::Domain(format!("json encode failed: {e}")))?;
    println!("{s}");
    Ok(())
}

// ---- inspect --------------------------------------------------------

fn cmd_inspect(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let as_json = take_bool_flag(&mut args, "--json");
    let pos = positionals(args);
    let path = pos
        .first()
        .ok_or_else(|| CliError::Usage("inspect: missing path argument".into()))?;
    let report = load_report(path)?;
    if as_json {
        emit_json(&report)
    } else {
        let md = render_inspect(&report);
        print!("{md}");
        Ok(())
    }
}

fn cmd_dump_summary(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("dump-summary: missing path argument".into()))?;
    let bytes = std::fs::read(path)?;
    let trace = IterDumpTrace::from_bytes(&bytes).map_err(|e| CliError::Parse(e.to_string()))?;
    let mut out = String::new();
    use std::fmt::Write as _;
    writeln!(out, "# POUNCEIT v{} trace", trace.header.format_version).ok();
    writeln!(out).ok();
    writeln!(
        out,
        "- **name**: `{}`",
        trace.header.name.replace('`', "\u{02CB}"),
    )
    .ok();
    writeln!(
        out,
        "- **n** (variables): {}, **m** (constraints): {}",
        trace.header.n, trace.header.m,
    )
    .ok();
    writeln!(out, "- **records**: {}", trace.records.len()).ok();
    writeln!(out).ok();
    writeln!(out, "| iter | mu | inf_pr | inf_du | α_pr | α_du | f |").ok();
    writeln!(out, "|---|---|---|---|---|---|---|").ok();
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
                writeln!(out, "| ... | | | | | | |").ok();
            }
        }
        let r = &trace.records[i];
        writeln!(
            out,
            "| {} | {:.2e} | {:.2e} | {:.2e} | {:.3} | {:.3} | {:.6e} |",
            r.iter, r.mu, r.inf_pr, r.inf_du, r.alpha_pr, r.alpha_du, r.f,
        )
        .ok();
        last = Some(i);
    }
    print!("{out}");
    Ok(())
}

// ---- solve-report tools ---------------------------------------------

fn cmd_summary(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("summary: missing path argument".into()))?;
    let report = load_report(path)?;
    emit_json(&analysis::summarize(&report))
}

fn cmd_diagnose(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("diagnose: missing path argument".into()))?;
    let report = load_report(path)?;
    let findings = analysis::diagnose(&report);
    let n = findings.len();
    let body = serde_json::json!({"findings": findings, "n_findings": n});
    emit_json(&body)
}

fn cmd_find_stalls(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let min_window = take_flag(&mut args, "--min-window")?
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| CliError::Usage(format!("--min-window: {e}")))?
        .unwrap_or(5);
    if min_window < analysis::MIN_STALL_WINDOW {
        return Err(CliError::Usage(format!(
            "--min-window must be >= {} (a stall spans at least two consecutive \
             iterations); got {min_window}",
            analysis::MIN_STALL_WINDOW
        )));
    }
    let max_progress = take_flag(&mut args, "--max-progress")?
        .map(|s| s.parse::<f64>())
        .transpose()
        .map_err(|e| CliError::Usage(format!("--max-progress: {e}")))?
        .unwrap_or(0.3);
    let pos = positionals(args);
    let path = pos
        .first()
        .ok_or_else(|| CliError::Usage("find-stalls: missing path argument".into()))?;
    let report = load_report(path)?;
    let windows = analysis::find_stalls_with(&report, min_window, max_progress);
    let n = windows.len();
    let body = serde_json::json!({"windows": windows, "count": n});
    emit_json(&body)
}

fn cmd_convergence_trace(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let columns = take_flag(&mut args, "--columns")?;
    let pos = positionals(args);
    let path = pos
        .first()
        .ok_or_else(|| CliError::Usage("convergence-trace: missing path argument".into()))?;
    let report = load_report(path)?;
    let full = analysis::convergence_trace(&report);
    let full_json =
        serde_json::to_value(&full).map_err(|e| CliError::Domain(format!("encode trace: {e}")))?;
    let Some(cols_spec) = columns else {
        return emit_json(&full_json);
    };
    let want: Vec<&str> = cols_spec.split(',').map(|s| s.trim()).collect();
    let obj = full_json
        .as_object()
        .ok_or_else(|| CliError::Domain("convergence trace was not an object".into()))?;
    let mut filtered = serde_json::Map::new();
    let mut unknown: Vec<String> = Vec::new();
    for col in want {
        match obj.get(col) {
            Some(v) => {
                filtered.insert(col.to_string(), v.clone());
            }
            None => unknown.push(col.to_string()),
        }
    }
    if !unknown.is_empty() {
        return Err(CliError::Usage(format!(
            "unknown trace column(s): {unknown:?}. valid: {:?}",
            obj.keys().collect::<Vec<_>>(),
        )));
    }
    emit_json(&serde_json::Value::Object(filtered))
}

fn cmd_get_iterate(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("get-iterate: missing path argument".into()))?;
    let k_str = args
        .get(1)
        .ok_or_else(|| CliError::Usage("get-iterate: missing iter index".into()))?;
    let k: usize = k_str
        .parse()
        .map_err(|e| CliError::Usage(format!("iter index: {e}")))?;
    let report = load_report(path)?;
    let iter = analysis::get_iterate(&report, k).map_err(|e| CliError::Domain(e.to_string()))?;
    emit_json(&iter)
}

fn cmd_restoration_windows(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("restoration-windows: missing path argument".into()))?;
    let report = load_report(path)?;
    let windows = analysis::restoration_windows(&report);
    let n = windows.len();
    let body = serde_json::json!({"windows": windows, "count": n});
    emit_json(&body)
}

fn cmd_compare(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let labels_csv = take_flag(&mut args, "--labels")?;
    let pos = positionals(args);
    if pos.is_empty() {
        return Err(CliError::Usage(
            "compare: at least one report path required".into(),
        ));
    }
    let labels: Vec<String> = match labels_csv {
        Some(csv) => csv.split(',').map(|s| s.trim().to_string()).collect(),
        None => pos.clone(),
    };
    if labels.len() != pos.len() {
        return Err(CliError::Usage(format!(
            "compare: labels count ({}) does not match paths count ({})",
            labels.len(),
            pos.len(),
        )));
    }
    let mut reports: Vec<(String, SolveReport)> = Vec::with_capacity(pos.len());
    for (label, path) in labels.iter().zip(pos.iter()) {
        let r = load_report(path)?;
        reports.push((label.clone(), r));
    }
    let rows = analysis::compare_runs(reports.iter().map(|(l, r)| (l.as_str(), r)));
    let n = rows.len();
    let body = serde_json::json!({"rows": rows, "n_runs": n});
    emit_json(&body)
}

fn cmd_linear_solver_summary(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("linear-solver-summary: missing path argument".into()))?;
    let report = load_report(path)?;
    let summary = &report.linear_solver;
    let body = serde_json::json!({
        "available": summary.is_some(),
        "summary": summary,
    });
    emit_json(&body)
}

// ---- glossary / citations -------------------------------------------

fn cmd_explain(args: &[String]) -> Result<(), CliError> {
    let term = args
        .first()
        .ok_or_else(|| CliError::Usage("explain: missing term argument".into()))?;
    let exp = glossary::explain(term);
    emit_json(&exp)
}

fn cmd_citations(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let topic = take_flag(&mut args, "--topic")?;
    let key = take_flag(&mut args, "--key")?;
    if topic.is_some() && key.is_some() {
        return Err(CliError::Usage(
            "citations: specify at most one of --topic or --key".into(),
        ));
    }
    if let Some(k) = key {
        let Some(c) = glossary::citation_by_key(&k) else {
            return Err(CliError::Domain(format!("unknown citation key {k:?}")));
        };
        return emit_json(c);
    }
    if let Some(t) = topic {
        let Some(keys) = glossary::topic_keys(&t) else {
            return Err(CliError::Domain(format!(
                "unknown topic {t:?}. valid: {:?}",
                glossary::all_topics(),
            )));
        };
        let entries: Vec<serde_json::Value> = keys
            .iter()
            .map(|k| match glossary::citation_by_key(k) {
                Some(c) => serde_json::to_value(c).unwrap_or(serde_json::Value::Null),
                None => serde_json::json!({"key": k, "missing": true}),
            })
            .collect();
        return emit_json(&serde_json::json!({"topic": t, "entries": entries}));
    }
    // No args: list topics + all loaded keys.
    let topics: serde_json::Map<String, serde_json::Value> = glossary::TOPICS
        .iter()
        .map(|(t, keys)| {
            (
                (*t).to_string(),
                serde_json::Value::Array(
                    keys.iter()
                        .map(|k| serde_json::Value::String((*k).to_string()))
                        .collect(),
                ),
            )
        })
        .collect();
    let entries = glossary::all_citations();
    let body = serde_json::json!({
        "topics": topics,
        "n_entries_loaded": entries.len(),
        "entries": entries,
    });
    emit_json(&body)
}

// ---- pre-flight: NL / builtin / GMS / lst ---------------------------

fn cmd_analyze_nl(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let builtin = take_flag(&mut args, "--builtin")?;
    let pos = positionals(args);

    if let Some(name) = builtin {
        if !pos.is_empty() {
            return Err(CliError::Usage(
                "analyze-nl: pass either --builtin NAME or a path, not both".into(),
            ));
        }
        let a = preflight::analyze_builtin(&name).map_err(CliError::Domain)?;
        return emit_json(&a);
    }

    let path = pos.first().ok_or_else(|| {
        CliError::Usage("analyze-nl: specify --builtin NAME or pass an .nl path".into())
    })?;
    let bytes = std::fs::read(path).map_err(|e| {
        CliError::Io(std::io::Error::new(
            e.kind(),
            format!("could not read {path:?}: {e}"),
        ))
    })?;
    let header = preflight::parse_nl_header(&bytes);
    let analysis = preflight::analyze_nl(path, header);
    emit_json(&analysis)
}

fn cmd_analyze_gms(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("analyze-gms: missing path argument".into()))?;
    let text = std::fs::read_to_string(path).map_err(|e| {
        CliError::Io(std::io::Error::new(
            e.kind(),
            format!("could not read {path:?}: {e}"),
        ))
    })?;
    let analysis = preflight::analyze_gms(path, &text);
    emit_json(&analysis)
}

fn cmd_parse_gams_listing(args: &[String]) -> Result<(), CliError> {
    let path = args
        .first()
        .ok_or_else(|| CliError::Usage("parse-gams-listing: missing .lst path argument".into()))?;
    let text = std::fs::read_to_string(path).map_err(|e| {
        CliError::Io(std::io::Error::new(
            e.kind(),
            format!("could not read {path:?}: {e}"),
        ))
    })?;
    let summary = preflight::parse_lst_summary(&text);
    let body = serde_json::json!({"path": path, "summary": summary});
    emit_json(&body)
}

// ---- repository inventory tools -------------------------------------

fn cmd_list_gams(args: &[String]) -> Result<(), CliError> {
    let mut args = args.to_vec();
    let root = take_flag(&mut args, "--root")?;
    let suite = take_flag(&mut args, "--suite")?;
    let limit: usize = take_flag(&mut args, "--limit")?
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| CliError::Usage(format!("--limit: {e}")))?
        .unwrap_or(50);
    let offset: usize = take_flag(&mut args, "--offset")?
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| CliError::Usage(format!("--offset: {e}")))?
        .unwrap_or(0);

    let root_path = match root {
        Some(r) => std::path::PathBuf::from(r),
        None => find_repo_root().ok_or_else(|| {
            CliError::Usage(
                "list-gams: could not locate pounce repo root; pass --root <path>".into(),
            )
        })?,
    };
    let gams_dir = root_path.join("gams");

    let suites = [
        "globallib.gms",
        "mittelmann.gms",
        "princetonlib.gms",
        "powerflow.gms",
    ];

    if let Some(name) = suite {
        let sroot = match name.as_str() {
            "examples" => gams_dir.join("examples"),
            "smoke" => gams_dir.clone(),
            s if suites.contains(&s) => gams_dir.join("nlpbench").join("instances").join(s),
            other => {
                let mut valid: Vec<String> = suites.iter().map(|s| (*s).to_string()).collect();
                valid.push("examples".into());
                valid.push("smoke".into());
                return Err(CliError::Usage(format!(
                    "unknown suite {other:?}; valid: {valid:?}",
                )));
            }
        };
        let mut files: Vec<String> = if name == "smoke" {
            let f = gams_dir.join("test_hs071.gms");
            if f.exists() {
                vec![f.to_string_lossy().into()]
            } else {
                vec![]
            }
        } else if sroot.is_dir() {
            collect_gms(&sroot)
        } else {
            Vec::new()
        };
        files.sort();
        let total = files.len();
        let slice: Vec<&String> = files.iter().skip(offset).take(limit).collect();
        let body = serde_json::json!({
            "suite": name,
            "root": sroot.to_string_lossy(),
            "count": total,
            "limit": limit,
            "offset": offset,
            "files": slice,
        });
        return emit_json(&body);
    }

    // No --suite: summary across all of them.
    let mut entries: Vec<serde_json::Value> = Vec::new();
    let mut total = 0usize;
    for s in suites.iter() {
        let d = gams_dir.join("nlpbench").join("instances").join(s);
        let n = if d.is_dir() { collect_gms(&d).len() } else { 0 };
        total += n;
        entries.push(serde_json::json!({"name": s, "count": n, "root": d.to_string_lossy()}));
    }
    let ex = gams_dir.join("examples");
    let n_ex = if ex.is_dir() {
        collect_gms(&ex).len()
    } else {
        0
    };
    total += n_ex;
    entries
        .push(serde_json::json!({"name": "examples", "count": n_ex, "root": ex.to_string_lossy()}));
    let smoke = gams_dir.join("test_hs071.gms");
    let n_smoke = if smoke.exists() { 1 } else { 0 };
    total += n_smoke;
    entries.push(
        serde_json::json!({"name": "smoke", "count": n_smoke, "root": gams_dir.to_string_lossy()}),
    );

    emit_json(&serde_json::json!({"suites": entries, "total": total}))
}

fn cmd_list_builtins(_args: &[String]) -> Result<(), CliError> {
    let all = preflight::all_builtins();
    emit_json(&all)
}

fn collect_gms(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "gms").unwrap_or(false) {
                out.push(p.to_string_lossy().into_owned());
            }
        }
    }
    out
}

/// Walk up from CWD looking for a directory that contains `Cargo.toml`
/// and `gams/` — the pounce repo root layout.
fn find_repo_root() -> Option<std::path::PathBuf> {
    let mut cwd = std::env::current_dir().ok()?;
    loop {
        if cwd.join("Cargo.toml").exists() && cwd.join("gams").is_dir() {
            return Some(cwd);
        }
        if !cwd.pop() {
            return None;
        }
    }
}

// ---- help -----------------------------------------------------------

fn print_help() {
    print!("{}", HELP_TEXT);
}

fn print_help_to_stderr() {
    eprint!("{}", HELP_TEXT);
}

const HELP_TEXT: &str = "pounce-studio — inspector for pounce solve reports + pre-flight tools

USAGE:
  pounce-studio <COMMAND> [ARGS]

REPORT POST-MORTEM (operate on a `pounce.solve-report/v1` JSON):
  summary <report>                          Headline summary as JSON
  diagnose <report>                         Common Ipopt-failure heuristics
  find-stalls <report>                      Stalled-progress windows
      [--min-window N] [--max-progress P]
  convergence-trace <report>                Per-iter trajectory (column-oriented)
      [--columns iter,inf_pr,...]
  get-iterate <report> <k>                  Full iter k record + log10 fields
  restoration-windows <report>              Restoration entry → exit cycles
  compare <r1> <r2> ...                     Side-by-side row per report
      [--labels A,B,C]
  linear-solver-summary <report>            FERAL backend post-mortem
  inspect <report> [--json]                 Markdown summary (default) or JSON
  dump-summary <trace>                      Markdown summary of a POUNCEIT trace

GLOSSARY / CITATIONS:
  explain <term>                            Define a column or finding code
  citations [--topic T] [--key K]           Curated paper references

PRE-FLIGHT:
  analyze-nl <path>                         AMPL .nl header + suggestions
  analyze-nl --builtin <NAME>               Builtin-problem metadata
  analyze-gms <path>                        GAMS .gms header + suggestions
  parse-gams-listing <path>                 GAMS .lst SOLVE SUMMARY block
  list-gams [--suite S] [--root DIR]        Enumerate bundled .gms instances
      [--limit N] [--offset M]
  list-builtins                             Names + classes of all builtin problems

OTHER:
  version | --version | -V                  Print crate version
  help    | --help    | -h                  Show this message

All analysis subcommands print pretty-printed JSON on stdout. Pipe into
`jq` for slicing. See `studio/skill/SKILL.md` for the Claude-skill that
wraps this binary for AI-assisted post-mortem.
";
