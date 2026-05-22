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
    /// `--json-output PATH` — when set, the binary writes a
    /// machine-readable JSON solve report to PATH after the solve
    /// completes. See [`crate::solve_report`] (pounce#8).
    pub json_output: Option<PathBuf>,
    /// `--json-detail summary|full` — controls how much detail the
    /// JSON report carries. Defaults to `Summary`. `Full` adds
    /// per-iteration history and suffix blocks; same scale as
    /// upstream's `print_level` but on the JSON side.
    pub json_detail: crate::solve_report::ReportDetail,
    /// `--sol-output PATH` — write an AMPL `.sol` solution file to
    /// PATH. When unset, a positional `.nl` input still gets a sibling
    /// `<stub>.sol` (the AMPL solver convention); `--no-sol` opts out
    /// of that default. Builtin problems have no stub, so they only
    /// produce a `.sol` when this flag is given explicitly.
    pub sol_output: Option<PathBuf>,
    /// `--no-sol` — suppress the default `<stub>.sol` write for `.nl`
    /// inputs.
    pub no_sol: bool,
    /// `-AMPL` — the AMPL solver-protocol flag. AMPL and Pyomo's ASL
    /// interface invoke a solver as `solver problem.nl -AMPL`. It needs
    /// no positional behavior (pounce already reads the `.nl` and
    /// writes `<stub>.sol`), but it does switch the process exit-code
    /// contract: in AMPL mode the termination is conveyed through the
    /// `.sol` file's `solve_result_num`, so the process exits 0 for any
    /// non-fatal solve outcome (limit reached, infeasible, etc.) rather
    /// than the non-zero code the plain CLI uses.
    pub ampl: bool,
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
    /// `--sens-boundcheck` — clamp the perturbed primal `x* + Δx` onto
    /// the declared `[x_l, x_u]` box after the sensitivity step. Only
    /// has effect when the `.nl` declares the sIPOPT suffixes. Mirrors
    /// upstream sIPOPT's `sens_boundcheck`.
    pub sens_boundcheck: bool,
    /// `--sens-bound-eps <eps>` — tolerance for `--sens-boundcheck`
    /// (default `1e-3`). Setting it also enables `--sens-boundcheck`.
    pub sens_bound_eps: f64,
    /// `--compute-red-hessian` — after the solve, compute the reduced
    /// Hessian over the variables tagged by the `red_hessian` integer
    /// var-suffix in the input `.nl`. Mirrors upstream sIPOPT's
    /// `compute_red_hessian`.
    pub compute_red_hessian: bool,
    /// `--rh-eigendecomp` — also compute the eigendecomposition of the
    /// reduced Hessian. Implies `--compute-red-hessian`. Mirrors
    /// upstream `rh_eigendecomp`.
    pub rh_eigendecomp: bool,
}

impl Args {
    pub fn usage() -> &'static str {
        "\
Usage: pounce [OPTIONS] [PATH] [SOL] [KEY=VALUE ...]

PATH is an AMPL .nl file (positional). Equivalent: --nl-file <path>.
SOL is an optional second positional naming the .sol output file
(equivalent to --sol-output <path>); the AMPL `solver in.nl out.sol`
convention.

When the .nl declares the sIPOPT suffixes (sens_state_1,
sens_state_value_1, sens_init_constr), pounce additionally runs the
post-optimal parametric sensitivity step and writes the perturbed
primal back into the .sol as a `sens_sol_state_1` suffix.

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
  --json-output <path>      write a JSON solve report to PATH after the solve
                            (pounce#8 — machine-readable, FAIR-aligned)
  --json-detail LEVEL       summary | full (default: summary). `full` adds
                            per-iteration history + suffix blocks.
  --sol-output <path>       write an AMPL .sol solution file to PATH.
                            A positional .nl input writes <stub>.sol
                            next to it by default (AMPL convention).
  --no-sol                  suppress the default <stub>.sol write
  --sens-boundcheck         clamp the perturbed primal x* + Δx onto the
                            declared [x_l, x_u] box (sIPOPT sens_boundcheck)
  --sens-bound-eps EPS      tolerance for --sens-boundcheck (default 1e-3;
                            setting it also enables --sens-boundcheck)
  --compute-red-hessian     compute the reduced Hessian over the variables
                            tagged by the `red_hessian` integer var-suffix
  --rh-eigendecomp          also compute the reduced-Hessian eigendecomp;
                            implies --compute-red-hessian
  --list-problems           print available built-in problems and exit
  -AMPL                     AMPL solver-protocol mode (for Pyomo / AMPL
                            drivers): convey termination via the .sol
                            file and exit 0 for non-fatal outcomes
  --help, -h                print this message and exit
  --version, -v, -V         print version and exit
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
        let mut json_output: Option<PathBuf> = None;
        let mut json_detail = crate::solve_report::ReportDetail::Summary;
        let mut sol_output: Option<PathBuf> = None;
        let mut no_sol = false;
        let mut ampl = false;
        let mut help = false;
        let mut version = false;
        let mut about = false;
        let mut list_problems = false;
        let mut dump_specs: Vec<(String, String)> = Vec::new();
        let mut dump_dir: Option<PathBuf> = None;
        let mut dump_format: Option<String> = None;
        let mut sens_boundcheck = false;
        let mut sens_bound_eps: f64 = 1e-3;
        let mut compute_red_hessian = false;
        let mut rh_eigendecomp = false;

        let mut it = argv.into_iter().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => help = true,
                "-v" | "-V" | "--version" => version = true,
                "--about" => about = true,
                // AMPL solver-protocol flag — see `Args::ampl`.
                "-AMPL" => ampl = true,
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
                "--json-output" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--json-output requires a value".to_string())?;
                    json_output = Some(PathBuf::from(v));
                }
                "--json-detail" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--json-detail requires a value".to_string())?;
                    json_detail = crate::solve_report::ReportDetail::parse(&v)?;
                }
                "--sol-output" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--sol-output requires a value".to_string())?;
                    sol_output = Some(PathBuf::from(v));
                }
                "--no-sol" => no_sol = true,
                "--sens-boundcheck" => sens_boundcheck = true,
                "--sens-bound-eps" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--sens-bound-eps requires a value".to_string())?;
                    sens_bound_eps = v
                        .parse::<f64>()
                        .map_err(|e| format!("--sens-bound-eps: {e}"))?;
                    sens_boundcheck = true;
                }
                "--compute-red-hessian" => compute_red_hessian = true,
                "--rh-eigendecomp" => {
                    rh_eigendecomp = true;
                    compute_red_hessian = true;
                }
                other if !other.starts_with('-') => {
                    // `key=value` forms an option pair (matches upstream
                    // ipopt CLI). Otherwise the first bare arg is the
                    // positional .nl path, and a second bare arg is the
                    // .sol output (AMPL `solver in.nl out.sol`).
                    if let Some((k, v)) = parse_kv(other) {
                        set_options.push((k, v));
                    } else if problem.is_none() {
                        problem = Some(ProblemSource::NlFile(PathBuf::from(other)));
                    } else if sol_output.is_none() {
                        sol_output = Some(PathBuf::from(other));
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
            let problem = problem.ok_or_else(|| {
                "missing problem: pass a positional .nl path, --nl-file, or --problem".to_string()
            })?;
            return Ok(Self {
                problem,
                options_file,
                set_options,
                json_output,
                json_detail,
                sol_output,
                no_sol,
                ampl,
                help,
                version,
                about,
                dump_specs,
                dump_dir,
                dump_format,
                sens_boundcheck,
                sens_bound_eps,
                compute_red_hessian,
                rh_eigendecomp,
            });
        }

        Ok(Self {
            problem: ProblemSource::Builtin(String::new()),
            options_file,
            set_options,
            json_output,
            json_detail,
            sol_output,
            no_sol,
            ampl,
            help,
            version,
            about,
            dump_specs,
            dump_dir,
            dump_format,
            sens_boundcheck,
            sens_bound_eps,
            compute_red_hessian,
            rh_eigendecomp,
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
        assert!(Args::parse_argv(argv(&["-v"])).unwrap().version);
        assert!(Args::parse_argv(argv(&["-V"])).unwrap().version);
        assert!(Args::parse_argv(argv(&["--version"])).unwrap().version);
    }

    #[test]
    fn ampl_flag_sets_mode_and_keeps_positional() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "-AMPL"])).unwrap();
        assert!(a.ampl);
        match a.problem {
            ProblemSource::NlFile(p) => assert_eq!(p.to_str(), Some("/tmp/foo.nl")),
            _ => panic!("expected positional .nl"),
        }
    }

    #[test]
    fn ampl_flag_defaults_off() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl"])).unwrap();
        assert!(!a.ampl);
    }

    #[test]
    fn ampl_flag_with_options() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "-AMPL", "max_iter=500"])).unwrap();
        assert!(a.ampl);
        assert_eq!(a.set_options, vec![("max_iter".into(), "500".into())]);
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
    fn sol_output_captured() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--sol-output", "/tmp/out.sol"])).unwrap();
        assert_eq!(a.sol_output.unwrap().to_str(), Some("/tmp/out.sol"));
        assert!(!a.no_sol);
    }

    #[test]
    fn no_sol_flag() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--no-sol"])).unwrap();
        assert!(a.no_sol);
        assert!(a.sol_output.is_none());
    }

    #[test]
    fn sol_output_defaults_unset() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl"])).unwrap();
        assert!(a.sol_output.is_none());
        assert!(!a.no_sol);
    }

    #[test]
    fn sol_output_missing_value() {
        assert!(Args::parse_argv(argv(&["/tmp/foo.nl", "--sol-output"])).is_err());
    }

    #[test]
    fn second_positional_is_sol_output() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "/tmp/out.sol"])).unwrap();
        match a.problem {
            ProblemSource::NlFile(p) => assert_eq!(p.to_str(), Some("/tmp/foo.nl")),
            _ => panic!("expected positional .nl"),
        }
        assert_eq!(a.sol_output.unwrap().to_str(), Some("/tmp/out.sol"));
    }

    #[test]
    fn third_positional_is_an_error() {
        assert!(Args::parse_argv(argv(&["/tmp/a.nl", "/tmp/b.sol", "/tmp/c"])).is_err());
    }

    #[test]
    fn sens_flags_default_off() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl"])).unwrap();
        assert!(!a.sens_boundcheck);
        assert!(!a.compute_red_hessian);
        assert!(!a.rh_eigendecomp);
        assert_eq!(a.sens_bound_eps, 1e-3);
    }

    #[test]
    fn sens_boundcheck_flag() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--sens-boundcheck"])).unwrap();
        assert!(a.sens_boundcheck);
    }

    #[test]
    fn sens_bound_eps_sets_value_and_enables_boundcheck() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--sens-bound-eps", "1e-6"])).unwrap();
        assert_eq!(a.sens_bound_eps, 1e-6);
        assert!(a.sens_boundcheck);
    }

    #[test]
    fn rh_eigendecomp_implies_compute_red_hessian() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--rh-eigendecomp"])).unwrap();
        assert!(a.rh_eigendecomp);
        assert!(a.compute_red_hessian);
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
