//! Argv parser for the `pounce` binary. Tiny hand-rolled parser so we
//! avoid pulling in `clap` (and its 100k LOC dependency tree).

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum ProblemSource {
    Builtin(String),
    NlFile(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Args {
    pub problem: ProblemSource,
    pub options_file: Option<PathBuf>,
    /// `key=value` options collected from the command line. Forwarded to
    /// the application's `OptionsList` after the options-file load (so
    /// CLI args override file values), mirroring upstream ipopt's
    /// `ipopt problem.nl print_level=8 ...` convention.
    pub set_options: Vec<(String, String)>,
    pub help: bool,
    pub version: bool,
    /// `--about`: print build metadata, compiled-in features, available
    /// linear solvers, and runtime paths. Used for bug reports.
    pub about: bool,
    /// `--dump <cat>[:<iter-spec>]`, repeatable. Each entry asks the
    /// solver to dump one diagnostic category at the specified iter
    /// range (`all`, `N`, `N-M`, `N-`, `-M`); omitting the spec is
    /// equivalent to `:all`. Forwarded to
    /// [`pounce_common::diagnostics::DiagnosticsConfig`].
    pub dump_specs: Vec<(String, String)>,
    /// `--dump-dir <path>`: override the dump root. Defaults to
    /// `./pounce-dump-<unix-secs>`, picked at solve-start time.
    pub dump_dir: Option<PathBuf>,
    /// `--dump-format <fmt>`: dump file format. Currently only `jsonl`.
    pub dump_format: Option<String>,
}

impl Args {
    pub fn usage() -> &'static str {
        "\
Usage: pounce [OPTIONS] [PATH] [KEY=VALUE ...]

PATH is an AMPL .nl file (positional). Equivalent: --nl-file <path>.

Trailing KEY=VALUE pairs are forwarded to the solver's OptionsList
(same syntax/semantics as the ipopt CLI). They override values loaded
from --options-file. Examples:

  pounce problem.nl print_level=8
  pounce problem.nl max_iter=500 tol=1e-10 linear_solver=ma57

Required (one of):
  PATH                      positional .nl file to solve
  --nl-file <path>          same, as a flag
  --problem <name>          solve a built-in test problem

Options:
  --options-file <path>     read solver options from an ipopt.opt-format file
  --list-problems           print available built-in problems and exit
  --help, -h                print this message and exit
  --version, -V             print version and exit
  --about                   print version, build info, features,
                            linear solvers, and runtime paths
  --dump <cat>[:<spec>]     dump diagnostic category to per-iter files.
                            Repeatable. Categories: kkt, iterate, step,
                            mu, ls, resto, convergence, timing.
                            Iter-spec grammar: all | N | N-M | N- | -M
                            (default: all). Examples:
                              --dump kkt:5
                              --dump kkt:2-10 --dump iterate:all
  --dump-dir <path>         override dump root (default ./pounce-dump-<ts>)
  --dump-format <fmt>       dump format (default: jsonl)
"
    }

    pub fn parse_argv(argv: Vec<String>) -> Result<Self, String> {
        let mut problem: Option<ProblemSource> = None;
        let mut options_file: Option<PathBuf> = None;
        let mut set_options: Vec<(String, String)> = Vec::new();
        let mut help = false;
        let mut version = false;
        let mut about = false;
        let mut list_problems = false;
        let mut dump_specs: Vec<(String, String)> = Vec::new();
        let mut dump_dir: Option<PathBuf> = None;
        let mut dump_format: Option<String> = None;

        let mut it = argv.into_iter().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => help = true,
                "-V" | "--version" => version = true,
                "--about" => about = true,
                "--list-problems" => list_problems = true,
                "--problem" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--problem requires a value".to_string())?;
                    problem = Some(ProblemSource::Builtin(v));
                }
                "--nl-file" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--nl-file requires a value".to_string())?;
                    problem = Some(ProblemSource::NlFile(PathBuf::from(v)));
                }
                "--options-file" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--options-file requires a value".to_string())?;
                    options_file = Some(PathBuf::from(v));
                }
                "--dump" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--dump requires a value (cat[:spec])".to_string())?;
                    let (cat, spec) = match v.split_once(':') {
                        Some((c, s)) => (c.to_string(), s.to_string()),
                        None => (v, "all".to_string()),
                    };
                    dump_specs.push((cat, spec));
                }
                "--dump-dir" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--dump-dir requires a value".to_string())?;
                    dump_dir = Some(PathBuf::from(v));
                }
                "--dump-format" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--dump-format requires a value".to_string())?;
                    dump_format = Some(v);
                }
                other if !other.starts_with('-') => {
                    // `key=value` forms an option pair (matches upstream
                    // ipopt CLI). Otherwise it's the positional .nl path.
                    if let Some((k, v)) = parse_kv(other) {
                        set_options.push((k, v));
                    } else if problem.is_none() {
                        problem = Some(ProblemSource::NlFile(PathBuf::from(other)));
                    } else {
                        return Err(format!(
                            "unexpected positional argument '{other}' (expected KEY=VALUE)"
                        ));
                    }
                }
                other => return Err(format!("unrecognized argument '{other}'")),
            }
        }

        if list_problems {
            println!("{}", crate::builtin::list().join("\n"));
            std::process::exit(0);
        }

        if !help && !version && !about {
            let problem = problem
                .ok_or_else(|| "missing problem: pass a positional .nl path, --nl-file, or --problem".to_string())?;
            return Ok(Self {
                problem,
                options_file,
                set_options,
                help,
                version,
                about,
                dump_specs,
                dump_dir,
                dump_format,
            });
        }

        Ok(Self {
            problem: ProblemSource::Builtin(String::new()),
            options_file,
            set_options,
            help,
            version,
            about,
            dump_specs,
            dump_dir,
            dump_format,
        })
    }
}

/// Parse `key=value` (or `key:=value`, ipopt-compatible). Returns
/// `None` if the token does not contain `=`. Whitespace around the
/// separator is trimmed; empty key or value yields `None`.
fn parse_kv(s: &str) -> Option<(String, String)> {
    let (k, v) = s.split_once('=')?;
    let k = k.trim().trim_end_matches(':');
    let v = v.trim();
    if k.is_empty() || v.is_empty() {
        return None;
    }
    Some((k.to_string(), v.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("pounce")
            .chain(args.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn help_short_and_long() {
        assert!(Args::parse_argv(argv(&["-h"])).unwrap().help);
        assert!(Args::parse_argv(argv(&["--help"])).unwrap().help);
    }

    #[test]
    fn version_short_and_long() {
        assert!(Args::parse_argv(argv(&["-V"])).unwrap().version);
        assert!(Args::parse_argv(argv(&["--version"])).unwrap().version);
    }

    #[test]
    fn about_flag_does_not_require_problem() {
        let a = Args::parse_argv(argv(&["--about"])).unwrap();
        assert!(a.about);
    }

    #[test]
    fn problem_flag_captures_name() {
        let a = Args::parse_argv(argv(&["--problem", "rosenbrock"])).unwrap();
        match a.problem {
            ProblemSource::Builtin(s) => assert_eq!(s, "rosenbrock"),
            _ => panic!("expected builtin"),
        }
    }

    #[test]
    fn nl_file_captured() {
        let a = Args::parse_argv(argv(&["--nl-file", "/tmp/foo.nl"])).unwrap();
        match a.problem {
            ProblemSource::NlFile(p) => assert_eq!(p.to_str(), Some("/tmp/foo.nl")),
            _ => panic!("expected nl file"),
        }
    }

    #[test]
    fn positional_nl_path() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl"])).unwrap();
        match a.problem {
            ProblemSource::NlFile(p) => assert_eq!(p.to_str(), Some("/tmp/foo.nl")),
            _ => panic!("expected positional .nl"),
        }
    }

    #[test]
    fn positional_with_options_file() {
        let a = Args::parse_argv(argv(&["--options-file", "ipopt.opt", "/tmp/foo.nl"])).unwrap();
        match a.problem {
            ProblemSource::NlFile(p) => assert_eq!(p.to_str(), Some("/tmp/foo.nl")),
            _ => panic!("expected positional .nl"),
        }
        assert_eq!(a.options_file.unwrap().to_str(), Some("ipopt.opt"));
    }

    #[test]
    fn options_file_captured() {
        let a = Args::parse_argv(argv(&["--problem", "x", "--options-file", "ipopt.opt"])).unwrap();
        assert_eq!(a.options_file.unwrap().to_str(), Some("ipopt.opt"));
    }

    #[test]
    fn missing_value_for_flag() {
        assert!(Args::parse_argv(argv(&["--problem"])).is_err());
    }

    #[test]
    fn missing_problem() {
        assert!(Args::parse_argv(argv(&[])).is_err());
    }

    #[test]
    fn unknown_arg() {
        assert!(Args::parse_argv(argv(&["--bogus"])).is_err());
    }

    #[test]
    fn key_value_options_collected() {
        let a = Args::parse_argv(argv(&[
            "/tmp/foo.nl",
            "print_level=8",
            "max_iter=500",
            "tol=1e-10",
        ]))
        .unwrap();
        assert_eq!(
            a.set_options,
            vec![
                ("print_level".into(), "8".into()),
                ("max_iter".into(), "500".into()),
                ("tol".into(), "1e-10".into()),
            ]
        );
    }

    #[test]
    fn key_value_before_path() {
        let a = Args::parse_argv(argv(&["print_level=8", "/tmp/foo.nl"])).unwrap();
        match a.problem {
            ProblemSource::NlFile(p) => assert_eq!(p.to_str(), Some("/tmp/foo.nl")),
            _ => panic!("expected positional .nl"),
        }
        assert_eq!(a.set_options, vec![("print_level".into(), "8".into())]);
    }

    #[test]
    fn dump_flag_captures_cat_and_spec() {
        let a = Args::parse_argv(argv(&[
            "--problem",
            "x",
            "--dump",
            "kkt:2-10",
            "--dump",
            "iterate",
        ]))
        .unwrap();
        assert_eq!(
            a.dump_specs,
            vec![
                ("kkt".into(), "2-10".into()),
                ("iterate".into(), "all".into()),
            ]
        );
    }

    #[test]
    fn dump_dir_and_format_captured() {
        let a = Args::parse_argv(argv(&[
            "--problem",
            "x",
            "--dump",
            "kkt",
            "--dump-dir",
            "/tmp/d",
            "--dump-format",
            "jsonl",
        ]))
        .unwrap();
        assert_eq!(a.dump_dir.unwrap().to_str(), Some("/tmp/d"));
        assert_eq!(a.dump_format.as_deref(), Some("jsonl"));
    }

    #[test]
    fn parse_kv_basic() {
        assert_eq!(
            parse_kv("print_level=8"),
            Some(("print_level".into(), "8".into()))
        );
        assert_eq!(
            parse_kv("tol = 1e-10"),
            Some(("tol".into(), "1e-10".into()))
        );
        assert_eq!(parse_kv("plain_path.nl"), None);
        assert_eq!(parse_kv("=value"), None);
        assert_eq!(parse_kv("key="), None);
    }
}
