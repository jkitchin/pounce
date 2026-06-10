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
    /// `--cite [REPORT.json]`: print the citations a user should include
    /// when publishing pounce results, then exit. Always lists the static
    /// core (pounce itself + Wächter-Biegler). When a solve-report JSON
    /// path follows, adds solve-aware extras for features the run actually
    /// used (v1: the restoration phase). A terminal mode like `--about` —
    /// requires no problem.
    pub cite: bool,
    /// Optional solve-report path consumed by `--cite` (the immediately
    /// following argument, iff present and not another flag).
    pub cite_report: Option<PathBuf>,
    /// `--bibtex`: render `--cite` output as BibTeX instead of the human
    /// list. No effect without `--cite`.
    pub cite_bibtex: bool,
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
    /// `--debug` / `--debug-json` — drop into the interactive solver
    /// debugger at each iteration. `Repl` is the human line-oriented
    /// front end; `Json` speaks newline-delimited JSON so an LLM agent
    /// (or any program) can drive the loop. `None` disables it.
    pub debug: Option<DebugMode>,
    /// `--debug-on-error` — don't pause every iteration; instead run
    /// freely and only drop into the debugger at the terminal checkpoint
    /// *if the solve did not succeed*, for a post-mortem at the failing
    /// iterate. Implies `--debug` (REPL) when no `--debug*` mode is given.
    pub debug_on_error: bool,
    /// `--debug-on-interrupt` — run normally but install a Ctrl-C handler
    /// that drops into the debugger at the next iteration. No automatic
    /// pauses. Implies `--debug` (REPL) when no `--debug*` mode is given.
    pub debug_on_interrupt: bool,
    /// `--debug-script <file>` — run debugger commands from a file at the
    /// first pause (e.g. set breakpoints then `continue`). Implies
    /// `--debug` when no `--debug*` mode is given.
    pub debug_script: Option<PathBuf>,
    /// `--minima <method>` (or `--multistart`) — search for multiple local
    /// minima instead of a single solve. `None` keeps the default
    /// single-solve behaviour. See [`MinimaArgs`] for the strategy knobs.
    /// Mirrors `pounce.find_minima` (`python/pounce/_minima.py`).
    pub minima: Option<MinimaArgs>,
}

/// Global-search strategy for `--minima`. Mirrors the six methods of
/// `pounce.find_minima` (`python/pounce/_minima.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimaMethod {
    /// Random / Sobol' box sampling (restart).
    Multistart,
    /// Multi-Level Single Linkage clustering (Rinnooy Kan & Timmer 1987).
    Mlsl,
    /// Metropolis chain over minima (Wales & Doye 1997).
    Basinhopping,
    /// Repulsive Gaussian bumps (filled-function; Ge 1990).
    Flooding,
    /// Softened `1/‖x−x*‖^p` poles (deflation; Farrell et al. 2015).
    Deflation,
    /// Equal-height tunnel between descents (Levy & Montalvo 1985).
    Tunneling,
}

impl MinimaMethod {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "multistart" => Self::Multistart,
            "mlsl" => Self::Mlsl,
            "basinhopping" => Self::Basinhopping,
            "flooding" => Self::Flooding,
            "deflation" => Self::Deflation,
            "tunneling" => Self::Tunneling,
            other => {
                return Err(format!(
                    "unknown --minima method '{other}'; choose from \
                     multistart, mlsl, basinhopping, flooding, deflation, tunneling"
                ))
            }
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Multistart => "multistart",
            Self::Mlsl => "mlsl",
            Self::Basinhopping => "basinhopping",
            Self::Flooding => "flooding",
            Self::Deflation => "deflation",
            Self::Tunneling => "tunneling",
        }
    }
}

/// Parsed `--minima` configuration. Shared knobs have concrete defaults;
/// strategy-specific knobs are `Option`s resolved per-method in the driver
/// (so `"auto"` widths and curvature-based amplitudes match
/// `pounce.find_minima`). Field semantics mirror `_minima.py` exactly.
#[derive(Debug, Clone)]
pub struct MinimaArgs {
    pub method: MinimaMethod,
    /// Target: stop once this many distinct minima are found (default 10).
    pub n_minima: usize,
    /// Budget: hard cap on solver calls (default `8 * n_minima`).
    pub max_solves: Option<usize>,
    /// Give-up: stop after this many solves in a row that find nothing new.
    pub patience: usize,
    /// Two minima within this scaled distance are the same (default 1e-4).
    pub dedup: f64,
    /// Smallest Hessian eigenvalue tolerated by saddle rejection (1e-6).
    pub psd_tol: f64,
    /// Seed for the sampler / Sobol' scramble (default 0; reproducible).
    pub seed: u64,
    /// Use a scrambled Sobol' sequence for box sampling (default true).
    pub sobol: bool,
    // ---- strategy-specific knobs (None ⇒ per-method default) ----
    pub sigma: Option<f64>,
    pub sigma_frac: Option<f64>,
    pub amplitude: Option<f64>,
    pub amp_margin: Option<f64>,
    pub eta: Option<f64>,
    pub power: Option<f64>,
    pub soft: Option<f64>,
    pub length: Option<f64>,
    pub length_frac: Option<f64>,
    pub gamma: Option<f64>,
    pub samples_per_round: Option<usize>,
    pub step: Option<f64>,
    pub temperature: Option<f64>,
    pub restart_jitter: Option<f64>,
}

impl Default for MinimaArgs {
    fn default() -> Self {
        Self {
            // Matches `find_minima`'s default `method="deflation"`.
            method: MinimaMethod::Deflation,
            n_minima: 10,
            max_solves: None,
            patience: 8,
            dedup: 1e-4,
            psd_tol: 1e-6,
            seed: 0,
            sobol: true,
            sigma: None,
            sigma_frac: None,
            amplitude: None,
            amp_margin: None,
            eta: None,
            power: None,
            soft: None,
            length: None,
            length_frac: None,
            gamma: None,
            samples_per_round: None,
            step: None,
            temperature: None,
            restart_jitter: None,
        }
    }
}

/// Front end for the interactive solver debugger (`--debug*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugMode {
    /// Human-facing line REPL on stdin/stdout.
    Repl,
    /// Newline-delimited JSON protocol for an agent / program.
    Json,
}

impl Args {
    pub fn usage() -> &'static str {
        "\
Usage: pounce [OPTIONS] [PATH] [SOL] [KEY=VALUE ...]

PATH is an AMPL .nl file (positional). Equivalent: --nl-file <path>.
SOL is an optional second positional naming the .sol output file
(equivalent to --sol-output <path>); the AMPL `solver in.nl out.sol`
convention.

Subcommand:
  pounce verify <problem.nl> <claim.sol> [--feas-tol T] [--json-output P]
                            independently check that a .sol solution
                            satisfies the canonical .nl's constraints and
                            bounds, without trusting the solver/agent that
                            produced it. Exit 0 = feasible, 20 = violated.
                            Run `pounce verify --help` for details.

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
  --debug                   drop into the interactive solver debugger (a
                            pdb-for-the-IPM): pause each iteration to
                            inspect/mutate x, multipliers, mu, set
                            breakpoints, step/continue. Type `help` at
                            the pounce-dbg> prompt for commands.
  --debug-json              same loop, but speak newline-delimited JSON on
                            stdin/stdout so an LLM agent or program can drive
                            it. The first line is a self-describing `hello`
                            handshake (protocol version + every command,
                            event, checkpoint, metric, and capability), so a
                            client needs no out-of-band docs; each pause is one
                            JSON state object. Full spec: docs/src/debugger.md.
  --debug-on-error          don't pause every iteration; run freely and
                            drop into the debugger only if the solve fails,
                            for a post-mortem at the final iterate. Implies
                            --debug when no --debug* mode is given.
  --debug-on-interrupt      run normally but install a Ctrl-C handler that
                            drops into the debugger at the next iteration
                            (second Ctrl-C aborts). Implies --debug when no
                            --debug* mode is given.
  --debug-script <file>     run debugger commands from a file at the first
                            pause (e.g. set breakpoints then continue).
                            Implies --debug when no --debug* mode is given.
  --list-problems           print available built-in problems and exit
  -AMPL                     AMPL solver-protocol mode (for Pyomo / AMPL
                            drivers): convey termination via the .sol
                            file and exit 0 for non-fatal outcomes
  --help, -h                print this message and exit
  --version, -v, -V         print version and exit
  --about                   print version, build info, features,
                            linear solvers, and runtime paths
  --cite [REPORT.json]      print the papers to cite when publishing
                            pounce results, then exit. Always lists pounce
                            itself + Wächter-Biegler; pass a JSON solve
                            report (from --json-output) to also list papers
                            for features the run used (e.g. restoration).
  --bibtex                  with --cite, emit BibTeX instead of a text list
  --dump <cat>[:<spec>]     dump diagnostic category to per-iter files.
                            Repeatable. Categories: kkt, iterate(s), step,
                            mu, ls, resto, convergence, timing.
                            Iter-spec grammar: all | N | N-M | N- | -M
                            (default: all). The `iterates` category also
                            accepts a `:summary` (default) or `:full`
                            variant suffix and streams one JSONL row
                            per iter to <dump-dir>/iterates.jsonl. The
                            `kkt` category accepts `+L` / `+L+Lvals`
                            suffixes that add the LDLᵀ factor's
                            strict-lower pattern (and optional values)
                            plus the fill-reducing permutation to each
                            kkt_solve_NNN.jsonl record (feral backend
                            only; MA57 silently omits the L fields).
                            Examples:
                              --dump kkt:5
                              --dump kkt:2-10 --dump iterate:all
                              --dump kkt:5-10+L
                              --dump kkt:5-10+L+Lvals
                              --dump iterates:summary
                              --dump iterates:5-:full
  --dump-dir <path>         override dump root (default ./pounce-dump-<ts>)
  --dump-format <fmt>       dump format (default: jsonl)

Multistart / find-minima (search for several local minima, not one):
  --minima <method>         enable multistart with the given strategy:
                            multistart | mlsl | basinhopping |
                            flooding | deflation | tunneling
  --multistart              shorthand for --minima multistart
  --n-minima <N>            target number of distinct minima (default 10)
  --max-solves <N>          hard cap on solver calls (default 8*n_minima)
  --patience <N>            stop after N solves in a row that find nothing
                            new (default 8)
  --dedup <d>               minima within this per-dimension-scaled distance
                            are the same (default 1e-4)
  --psd-tol <t>             smallest Hessian eigenvalue tolerated by the
                            saddle-rejection check (default 1e-6)
  --seed <S>                seed for sampling / Sobol' scramble (default 0)
  --sobol / --no-sobol      use a scrambled Sobol' sequence for box
                            sampling (default: on)
  Strategy knobs (used only by the relevant --minima method; all optional):
    --sigma, --sigma-frac, --amplitude, --amp-margin   (flooding)
    --eta, --power, --soft, --length, --length-frac    (deflation/tunneling)
    --gamma, --samples-per-round                       (mlsl)
    --step, --temperature                              (basinhopping)
    --restart-jitter                                   (all restart fallbacks)

  When --minima is set, the global best minimum is written to <stub>.sol
  (the usual AMPL output), and the remaining minima, ranked by objective,
  to siblings <stub>.min001.sol, <stub>.min002.sol, ….  The JSON report
  (--json-output) gains a `minima` section listing every found minimum.
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
        let mut cite = false;
        let mut cite_report: Option<PathBuf> = None;
        let mut cite_bibtex = false;
        let mut list_problems = false;
        let mut dump_specs: Vec<(String, String)> = Vec::new();
        let mut dump_dir: Option<PathBuf> = None;
        let mut dump_format: Option<String> = None;
        let mut sens_boundcheck = false;
        let mut sens_bound_eps: f64 = 1e-3;
        let mut compute_red_hessian = false;
        let mut rh_eigendecomp = false;
        let mut debug: Option<DebugMode> = None;
        let mut debug_on_error = false;
        let mut debug_on_interrupt = false;
        let mut debug_script: Option<PathBuf> = None;
        let mut minima: Option<MinimaArgs> = None;
        // Global search is enabled ONLY by an explicit method selector
        // (`--minima <m>` / `--multistart`). The tuning knobs below
        // (`--seed`, `--patience`, …) populate the config but must not, on
        // their own, switch the run into multistart mode — track whether a
        // method was explicitly chosen and which lone knob (if any) was seen
        // so we can reject a knob-without-method invocation after parsing.
        let mut minima_method_explicit = false;
        let mut minima_knob: Option<&'static str> = None;

        let mut it = argv.into_iter().skip(1).peekable();
        // Shorthand: fetch the value for a flag that requires one.
        macro_rules! flag_val {
            ($flag:expr) => {
                it.next()
                    .ok_or_else(|| format!("{} requires a value", $flag))?
            };
        }
        // Parse a numeric value for a `--minima` knob, lazily creating the
        // config (default method = deflation, overridden by `--minima <m>`).
        macro_rules! minima_num {
            ($flag:expr, $ty:ty, $field:ident) => {{
                let v = flag_val!($flag);
                let parsed: $ty = v.parse().map_err(|e| format!("{}: {}", $flag, e))?;
                minima.get_or_insert_with(MinimaArgs::default).$field = parsed;
                if minima_knob.is_none() {
                    minima_knob = Some($flag);
                }
            }};
            ($flag:expr, $ty:ty, $field:ident, opt) => {{
                let v = flag_val!($flag);
                let parsed: $ty = v.parse().map_err(|e| format!("{}: {}", $flag, e))?;
                minima.get_or_insert_with(MinimaArgs::default).$field = Some(parsed);
                if minima_knob.is_none() {
                    minima_knob = Some($flag);
                }
            }};
        }
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => help = true,
                "-v" | "-V" | "--version" => version = true,
                "--about" => about = true,
                "--cite" => {
                    cite = true;
                    // Optional value: consume the next argument as the
                    // solve-report path only if it's present and is not
                    // itself a flag (so `--cite --bibtex` doesn't swallow
                    // the modifier, and bare `--cite` stays report-less).
                    if let Some(next) = it.peek() {
                        if !next.starts_with('-') {
                            cite_report = Some(PathBuf::from(it.next().unwrap()));
                        }
                    }
                }
                "--bibtex" => cite_bibtex = true,
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
                "--debug" => debug = Some(DebugMode::Repl),
                "--debug-json" => debug = Some(DebugMode::Json),
                "--debug-on-error" => debug_on_error = true,
                "--debug-on-interrupt" => debug_on_interrupt = true,
                "--debug-script" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "--debug-script requires a value".to_string())?;
                    debug_script = Some(PathBuf::from(v));
                }
                "--compute-red-hessian" => compute_red_hessian = true,
                "--rh-eigendecomp" => {
                    rh_eigendecomp = true;
                    compute_red_hessian = true;
                }
                // ---- multistart / find-minima (`--minima`) ----
                "--minima" => {
                    let v = flag_val!("--minima");
                    let method = MinimaMethod::parse(&v)?;
                    minima.get_or_insert_with(MinimaArgs::default).method = method;
                    minima_method_explicit = true;
                }
                "--multistart" => {
                    minima.get_or_insert_with(MinimaArgs::default).method =
                        MinimaMethod::Multistart;
                    minima_method_explicit = true;
                }
                "--n-minima" => minima_num!("--n-minima", usize, n_minima),
                "--max-solves" => minima_num!("--max-solves", usize, max_solves, opt),
                "--patience" => minima_num!("--patience", usize, patience),
                "--dedup" => minima_num!("--dedup", f64, dedup),
                "--psd-tol" => minima_num!("--psd-tol", f64, psd_tol),
                "--seed" => minima_num!("--seed", u64, seed),
                "--sobol" => {
                    minima.get_or_insert_with(MinimaArgs::default).sobol = true;
                    if minima_knob.is_none() {
                        minima_knob = Some("--sobol");
                    }
                }
                "--no-sobol" => {
                    minima.get_or_insert_with(MinimaArgs::default).sobol = false;
                    if minima_knob.is_none() {
                        minima_knob = Some("--no-sobol");
                    }
                }
                "--sigma" => minima_num!("--sigma", f64, sigma, opt),
                "--sigma-frac" => minima_num!("--sigma-frac", f64, sigma_frac, opt),
                "--amplitude" => minima_num!("--amplitude", f64, amplitude, opt),
                "--amp-margin" => minima_num!("--amp-margin", f64, amp_margin, opt),
                "--eta" => minima_num!("--eta", f64, eta, opt),
                "--power" => minima_num!("--power", f64, power, opt),
                "--soft" => minima_num!("--soft", f64, soft, opt),
                "--length" => minima_num!("--length", f64, length, opt),
                "--length-frac" => minima_num!("--length-frac", f64, length_frac, opt),
                "--gamma" => minima_num!("--gamma", f64, gamma, opt),
                "--samples-per-round" => {
                    minima_num!("--samples-per-round", usize, samples_per_round, opt)
                }
                "--step" => minima_num!("--step", f64, step, opt),
                "--temperature" => minima_num!("--temperature", f64, temperature, opt),
                "--restart-jitter" => minima_num!("--restart-jitter", f64, restart_jitter, opt),
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

        // `--debug-on-error` / `--debug-on-interrupt` / `--debug-script`
        // without an explicit mode imply the REPL.
        if (debug_on_error || debug_on_interrupt || debug_script.is_some()) && debug.is_none() {
            debug = Some(DebugMode::Repl);
        }

        if !help && !version && !about && !cite {
            // A `--minima` *tuning* knob on its own used to lazily create a
            // config and silently reroute the whole run into multistart
            // (deflation) mode — different console output and a dual-free
            // `.sol`. Global search must be opted into explicitly; reject a
            // lone knob with a message pointing at the method selectors.
            if let Some(knob) = minima_knob {
                if !minima_method_explicit {
                    return Err(format!(
                        "{knob} is a --minima tuning knob and has no effect on its own; \
                         enable global search with --minima <method> or --multistart"
                    ));
                }
            }
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
                cite,
                cite_report,
                cite_bibtex,
                dump_specs,
                dump_dir,
                dump_format,
                sens_boundcheck,
                sens_bound_eps,
                compute_red_hessian,
                rh_eigendecomp,
                debug,
                debug_on_error,
                debug_on_interrupt,
                debug_script,
                minima,
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
            cite,
            cite_report,
            cite_bibtex,
            dump_specs,
            dump_dir,
            dump_format,
            sens_boundcheck,
            sens_bound_eps,
            compute_red_hessian,
            rh_eigendecomp,
            debug,
            debug_on_error,
            debug_on_interrupt,
            debug_script,
            minima,
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
    fn cite_flag_alone_needs_no_problem_or_report() {
        let a = Args::parse_argv(argv(&["--cite"])).unwrap();
        assert!(a.cite);
        assert!(a.cite_report.is_none());
        assert!(!a.cite_bibtex);
    }

    #[test]
    fn cite_consumes_following_report_path() {
        let a = Args::parse_argv(argv(&["--cite", "run.json"])).unwrap();
        assert!(a.cite);
        assert_eq!(a.cite_report.unwrap().to_str(), Some("run.json"));
    }

    #[test]
    fn cite_does_not_swallow_a_following_flag() {
        let a = Args::parse_argv(argv(&["--cite", "--bibtex"])).unwrap();
        assert!(a.cite);
        assert!(a.cite_report.is_none());
        assert!(a.cite_bibtex);
    }

    #[test]
    fn cite_with_report_and_bibtex() {
        let a = Args::parse_argv(argv(&["--cite", "run.json", "--bibtex"])).unwrap();
        assert!(a.cite);
        assert_eq!(a.cite_report.unwrap().to_str(), Some("run.json"));
        assert!(a.cite_bibtex);
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
    fn minima_absent_by_default() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl"])).unwrap();
        assert!(a.minima.is_none());
    }

    #[test]
    fn minima_method_and_shared_knobs() {
        let a = Args::parse_argv(argv(&[
            "/tmp/foo.nl",
            "--minima",
            "flooding",
            "--n-minima",
            "5",
            "--max-solves",
            "42",
            "--patience",
            "3",
            "--dedup",
            "1e-2",
            "--psd-tol",
            "1e-8",
            "--seed",
            "7",
            "--no-sobol",
        ]))
        .unwrap();
        let m = a.minima.expect("minima parsed");
        assert_eq!(m.method, MinimaMethod::Flooding);
        assert_eq!(m.n_minima, 5);
        assert_eq!(m.max_solves, Some(42));
        assert_eq!(m.patience, 3);
        assert_eq!(m.dedup, 1e-2);
        assert_eq!(m.psd_tol, 1e-8);
        assert_eq!(m.seed, 7);
        assert!(!m.sobol);
    }

    #[test]
    fn multistart_shorthand_selects_multistart() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--multistart"])).unwrap();
        assert_eq!(a.minima.unwrap().method, MinimaMethod::Multistart);
    }

    #[test]
    fn minima_strategy_knobs_are_optional_and_parsed() {
        let a = Args::parse_argv(argv(&[
            "/tmp/foo.nl",
            "--minima",
            "deflation",
            "--eta",
            "2.5",
            "--power",
            "3",
            "--soft",
            "1e-4",
            "--length",
            "0.2",
            "--restart-jitter",
            "0.9",
        ]))
        .unwrap();
        let m = a.minima.unwrap();
        assert_eq!(m.method, MinimaMethod::Deflation);
        assert_eq!(m.eta, Some(2.5));
        assert_eq!(m.power, Some(3.0));
        assert_eq!(m.soft, Some(1e-4));
        assert_eq!(m.length, Some(0.2));
        assert_eq!(m.restart_jitter, Some(0.9));
        // Untouched knobs stay None.
        assert_eq!(m.sigma, None);
        assert_eq!(m.gamma, None);
    }

    #[test]
    fn minima_unknown_method_errors() {
        assert!(Args::parse_argv(argv(&["/tmp/foo.nl", "--minima", "nope"])).is_err());
    }

    /// Code-review 2026-06 item M14: a `--minima` tuning knob (`--seed`,
    /// `--patience`, `--no-sobol`, …) on its own used to lazily build a
    /// `MinimaArgs` and silently reroute the whole run into multistart
    /// (deflation) mode. It must now be rejected with a message pointing
    /// at the method selectors.
    #[test]
    fn lone_minima_knob_without_method_is_rejected() {
        let err = Args::parse_argv(argv(&["/tmp/foo.nl", "--seed", "42"]))
            .expect_err("lone --seed should be rejected");
        assert!(
            err.contains("--seed") && err.contains("--minima"),
            "error should name the knob and the method selectors; got: {err}"
        );
        // A no-value knob (`--no-sobol`) is rejected the same way.
        let err2 = Args::parse_argv(argv(&["/tmp/foo.nl", "--no-sobol"]))
            .expect_err("lone --no-sobol should be rejected");
        assert!(err2.contains("--no-sobol"), "got: {err2}");
        // And a lone knob does NOT leave the run in minima mode.
        assert!(Args::parse_argv(argv(&["/tmp/foo.nl", "--seed", "42"])).is_err());
    }

    /// The same knob is accepted once global search is explicitly enabled,
    /// regardless of flag order (knob before the method selector).
    #[test]
    fn minima_knob_with_explicit_method_is_accepted() {
        let a = Args::parse_argv(argv(&["/tmp/foo.nl", "--seed", "7", "--multistart"])).unwrap();
        let m = a.minima.expect("minima parsed");
        assert_eq!(m.method, MinimaMethod::Multistart);
        assert_eq!(m.seed, 7);
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
