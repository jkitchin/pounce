//! Interactive solver debugger front end — "pdb for the IPM".
//!
//! Implements [`pounce_algorithm::debug::DebugHook`]. The core fires us
//! at every checkpoint (today: the top of each outer iteration); we
//! pause, hand the user (or an agent) a command prompt, and apply
//! inspect / mutate / flow commands against the live [`DebugState`] before
//! returning [`DebugAction::Resume`] or [`DebugAction::Stop`].
//!
//! Two front ends share one command engine ([`SolverDebugger::dispatch`]):
//!
//!   * [`DebugMode::Repl`] — a human line REPL. Prompts and command
//!     output go to **stderr** so they never interleave with the
//!     solver's iteration table on stdout.
//!   * [`DebugMode::Json`] — a newline-delimited JSON protocol on
//!     stdin/stdout for an LLM agent, visual debugger, or any program.
//!     stdout is a *pure* protocol channel (the CLI routes the banner /
//!     problem stats / summary to stderr and forces `print_level 0`), so
//!     a GUI can consume it line-by-line. Session lifecycle:
//!       1. `{"event":"hello",…}`  — once, up front: protocol version,
//!          advertised capabilities, command / metric / block vocabulary.
//!       2. `{"event":"pause",…}`  — at each stop: iter, μ, residuals,
//!          dims, active breakpoints/conditions, and the firing `reason`.
//!       3. `{"event":"result",…}` — one per command, echoing the
//!          client's `request_id` for async correlation.
//!       4. `{"event":"terminated",…}` — emitted by the CLI after the
//!          solve, carrying the final status, iteration count, objective,
//!          and eval counts.
//!
//!     Commands may be a bare string or `{"cmd":…,"args":[…],"id":…}`.
//!
//! Flow / exit model: the debugger pauses at the *first* checkpoint (so
//! you get control at iter 0), then only when re-armed — by `step` (pause
//! next iteration), `break N` (pause at iter N), `break if …` (pause on a
//! condition), or `run N` (pause at iter ≥ N). Exit paths:
//!   * `continue` — run to the next breakpoint, else to completion.
//!   * `detach`   — stop pausing; run to completion.
//!   * `quit`     — stop now (surfaces as `UserRequestedStop`).
//!   * stdin EOF  — REPL (Ctrl-D) detaches and finishes; JSON (pipe
//!     closed → client gone) aborts the solve.
//!
//! Every non-kill path ends with a `terminated` event in JSON mode.

use crate::cli::DebugMode;
use pounce_algorithm::debug::{
    is_live_tolerance, DebugCtx, IterateSnapshot, ResidKind, Residual, BLOCK_NAMES,
};
use pounce_algorithm::debug_rank::{RankReport, RankRow};
use pounce_common::debug::{Checkpoint, DebugAction, DebugHook, DebugState};
use pounce_common::reg_options::{DefaultValue, OptionType, RegisteredOptions};
use pounce_nlp::ipopt_nlp::SplitNames;
use pounce_presolve::dulmage_mendelsohn::DulmageMendelsohnPartition;
use pounce_presolve::incidence::EqualityIncidence;
use pounce_presolve::matching::hopcroft_karp;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::{Context, Editor, Helper, Highlighter, Hinter, Validator};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::rc::Rc;

/// All command verbs, for `help` and `complete`.
const COMMANDS: &[&str] = &[
    "help",
    "info",
    "print",
    "step",
    "stepi",
    "continue",
    "run",
    "break",
    "tbreak",
    "watchpoint",
    "commands",
    "stop-at",
    "set",
    "get",
    "opt",
    "complete",
    "viz",
    "save",
    "load",
    "sweep",
    "multistart",
    "goto",
    "restart",
    "resolve",
    "ask",
    "watch",
    "diff",
    "diagnose",
    "source",
    "progress",
    "detach",
    "quit",
];

/// Events a user can `break on` (advertised in `hello.events`). Each is
/// derived from observable state at the relevant checkpoint.
const EVENTS: &[&str] = &[
    "resto_entered",
    "resto_exited",
    "regularized",
    "tiny_step",
    "ls_rejected",
    "mu_stalled",
    "nan",
];

/// μ is "stalled" once it has held (to relative tolerance) for this many
/// consecutive iterations.
const MU_STALL_ITERS: u32 = 3;

/// A data watchpoint: pause when a watched value changes by more than
/// `threshold` between iterations.
#[derive(Clone)]
struct WatchPoint {
    /// Source text, e.g. `x` or `x[3]`, for display.
    raw: String,
    block: String,
    idx: Option<usize>,
    threshold: f64,
    /// Last observed value(s); `None` until first seen.
    last: Option<Vec<f64>>,
}

/// Checkpoint names a user can `stop-at` (matches `Checkpoint::as_str`).
const CHECKPOINTS: &[&str] = &[
    "iter_start",
    "after_mu",
    "after_search_dir",
    "after_step",
    "step_rejected",
    "pre_restoration_entry",
    "post_restoration_exit",
    "terminated",
];

/// Request to re-run the solve from a captured point with new options.
/// Written by the `resolve` command into the shared [`RestartCell`] and
/// read by the CLI after the solve unwinds.
pub struct RestartRequest {
    /// Primal seed (the algorithm-space `x` at the time of `resolve`).
    /// Also drives `sweep` / `multistart`, where only `x` varies.
    pub seed_x: Vec<f64>,
    /// `set opt` edits staged during the session, to apply before re-solve.
    pub options: Vec<(String, String)>,
    /// Full primal-dual iterate (all 8 blocks + μ) captured at the pause,
    /// for a true warm `resolve` that continues from the current interior
    /// point. `None` for primal-only restarts (sweep / multistart). When
    /// present, the CLI installs it via `set_warm_start_iterate` and turns
    /// on `warm_start_init_point` / `warm_start_target_mu`.
    pub warm: Option<IterateSnapshot>,
}

/// Shared slot the debugger uses to hand a [`RestartRequest`] back to the
/// CLI's re-solve loop.
pub type RestartCell = Rc<std::cell::RefCell<Option<RestartRequest>>>;

/// One completed solve in a `sweep` / `multistart` run.
#[derive(Clone)]
struct SweepRecord {
    /// 0-based index in the sweep.
    idx: usize,
    /// The primal seed this solve started from.
    seed: Vec<f64>,
    /// Terminal `SolverReturn` (debug string).
    status: String,
    /// Final objective.
    objective: f64,
    /// Final primal infeasibility.
    inf_pr: f64,
    /// Iteration count at termination.
    iters: i32,
}

/// In-flight `sweep` state, carried across the CLI's re-solve loop (the
/// same debugger instance is re-armed each solve, so this persists). Each
/// queued seed is run as a full solve; the terminal checkpoint records the
/// outcome and launches the next.
struct SweepState {
    /// Starts not yet run.
    queue: VecDeque<Vec<f64>>,
    /// The seed of the solve currently running (recorded at its terminal).
    current: Option<Vec<f64>>,
    /// Completed solves, in order.
    records: Vec<SweepRecord>,
    /// Total starts requested (for progress display).
    total: usize,
    /// `pause_iters` to restore when the sweep finishes (a sweep runs each
    /// solve free, so it disables per-iteration pausing for the duration).
    saved_pause_iters: bool,
}

/// Cap on retained per-iteration snapshots (bounds rewind memory; oldest
/// are evicted first).
const SNAPSHOT_CAP: usize = 2000;

/// SolverReturn debug strings that count as a successful solve (so
/// `--debug-on-error` does *not* pause at the terminal checkpoint).
fn is_success_status(s: &str) -> bool {
    matches!(s, "Success" | "StopAtAcceptablePoint")
}

/// Parse a free-form numeric blob — values separated by commas, whitespace,
/// or newlines — into `f64`s (used by `load` and `sweep` for plain start
/// files). Errors on the first unparsable token.
fn parse_floats(s: &str) -> Result<Vec<f64>, String> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<f64>().map_err(|_| format!("bad number `{t}`")))
        .collect()
}

/// A `splitmix64` step — a tiny deterministic PRNG (no `rand` dependency).
/// Returns a uniform draw in `[-1, 1]` and advances the state.
fn splitmix_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Top 53 bits → [0,1), then map to [-1,1).
    ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Per-start PRNG seed: deterministic in `k` so a `multistart` reproduces.
fn seed_for(k: usize) -> u64 {
    0x9E37_79B9_7F4A_7C15u64
        ^ (k as u64)
            .wrapping_mul(0xD1B5_4A32_D192_ED03)
            .wrapping_add(1)
}

/// Sample `multistart` start `k`. Start 0 is the unperturbed `base` (so the
/// run always covers the current point). For `k ≥ 1`, each component is
/// drawn **uniformly in its box** `[loᵢ, hiᵢ]` when both bounds are finite;
/// where a bound is missing (`±∞`), it falls back to a relative jitter
/// `±rel·(|baseᵢ|+1)` around the base. Deterministic in `k`.
fn sample_start(base: &[f64], bounds: Option<(&[f64], &[f64])>, rel: f64, k: usize) -> Vec<f64> {
    if k == 0 {
        return base.to_vec();
    }
    let mut state = seed_for(k);
    base.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let unit = splitmix_unit(&mut state); // [-1, 1)
            if let Some((lo, hi)) = bounds {
                let (l, u) = (lo[i], hi[i]);
                if l.is_finite() && u.is_finite() && u > l {
                    // [-1,1) → [l, u).
                    return l + (u - l) * (unit * 0.5 + 0.5);
                }
            }
            xi + rel * (xi.abs() + 1.0) * unit
        })
        .collect()
}

/// `multistart` with no bounds — pure relative jitter around `base`.
#[cfg(test)]
fn jitter(base: &[f64], rel: f64, k: usize) -> Vec<f64> {
    sample_start(base, None, rel, k)
}

/// SIGINT → "break into the debugger at the next iteration". A first
/// Ctrl-C sets a pending flag the hook consumes at the next checkpoint;
/// a second Ctrl-C before that (or any Ctrl-C once detached) hard-exits,
/// preserving the usual "abort" escape hatch.
///
/// At a rustyline prompt the terminal is in raw mode, so Ctrl-C arrives
/// as input (handled as `Interrupted`) rather than as SIGINT — this handler
/// only fires while the solve is running. The prompt has its own analogous
/// double-tap: the first Ctrl-C cancels the line, a second quits the solve
/// (see [`SolverDebugger::on_prompt_interrupt`]).
pub mod interrupt {
    use nix::sys::signal::{self, SigHandler, Signal};
    use std::sync::atomic::{AtomicBool, Ordering};

    static PENDING: AtomicBool = AtomicBool::new(false);
    static INSTALLED: AtomicBool = AtomicBool::new(false);

    extern "C" fn handler(_sig: nix::libc::c_int) {
        // `swap` returns the previous value: if a break was already
        // pending and unconsumed, the user pressed Ctrl-C twice — abort.
        if PENDING.swap(true, Ordering::SeqCst) {
            // _exit is async-signal-safe; 130 = 128 + SIGINT.
            unsafe { nix::libc::_exit(130) };
        }
    }

    /// Install the handler once (idempotent). Call only when a debugger
    /// is active, so a normal run keeps default Ctrl-C behavior.
    pub fn install() {
        if INSTALLED.swap(true, Ordering::SeqCst) {
            return;
        }
        // SAFETY: `handler` only touches an atomic and `_exit`.
        unsafe {
            let _ = signal::signal(Signal::SIGINT, SigHandler::Handler(handler));
        }
    }

    /// Consume a pending break request (clears it).
    pub fn take() -> bool {
        PENDING.swap(false, Ordering::SeqCst)
    }

    /// Test-only: simulate a Ctrl-C without raising a real signal.
    #[cfg(test)]
    pub fn set_pending_for_test() {
        PENDING.store(true, Ordering::SeqCst);
    }
}

/// What to do after a command runs.
#[derive(Clone, Copy)]
enum Flow {
    /// Stay paused; keep reading commands.
    Stay,
    /// Resume solving.
    Resume,
    /// Stop the solve.
    Stop,
}

/// Outcome of one command: human lines + optional structured payload.
struct CmdOut {
    ok: bool,
    lines: Vec<String>,
    data: Option<serde_json::Value>,
    flow: Flow,
}

impl CmdOut {
    fn ok(lines: Vec<String>) -> Self {
        Self {
            ok: true,
            lines,
            data: None,
            flow: Flow::Stay,
        }
    }
    fn err(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            lines: vec![msg.into()],
            data: None,
            flow: Flow::Stay,
        }
    }
    fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
    fn flow(mut self, flow: Flow) -> Self {
        self.flow = flow;
        self
    }
}

/// Metric names accepted in `break if …` (and shown by `help`).
const METRICS: &[&str] = &["mu", "inf_pr", "inf_du", "obj", "err", "compl", "iter"];

/// A scalar the solver exposes for conditional breakpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Mu,
    InfPr,
    InfDu,
    Obj,
    NlpError,
    Compl,
    Iter,
}

impl Metric {
    fn parse(s: &str) -> Option<Metric> {
        Some(match s {
            "mu" => Metric::Mu,
            "inf_pr" => Metric::InfPr,
            "inf_du" => Metric::InfDu,
            "obj" | "objective" => Metric::Obj,
            "err" | "nlp_error" => Metric::NlpError,
            "compl" | "complementarity" => Metric::Compl,
            "iter" => Metric::Iter,
            _ => return None,
        })
    }
    fn eval(self, ctx: &dyn DebugState) -> f64 {
        match self {
            Metric::Mu => ctx.mu(),
            Metric::InfPr => ctx.inf_pr(),
            Metric::InfDu => ctx.inf_du(),
            Metric::Obj => ctx.objective(),
            Metric::NlpError => ctx.nlp_error(),
            Metric::Compl => ctx.complementarity(),
            Metric::Iter => ctx.iter() as f64,
        }
    }
}

/// Comparison operator for a conditional breakpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
}

impl CmpOp {
    fn eval(self, lhs: f64, rhs: f64) -> bool {
        match self {
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
            // Tolerant equality so float metrics aren't impossible to hit:
            // |lhs − rhs| ≤ 1e-12·max(1, |rhs|). Note this is relative for
            // large rhs but collapses to an absolute 1e-12 when rhs == 0, so
            // `obj==0` means "|obj| ≤ 1e-12" and `iter==N` is exact for the
            // integer-valued metrics.
            CmpOp::Eq => (lhs - rhs).abs() <= 1e-12 * rhs.abs().max(1.0),
        }
    }
}

/// A single comparison `metric op rhs`.
#[derive(Clone, Debug)]
struct Atom {
    metric: Metric,
    op: CmpOp,
    rhs: f64,
}

impl Atom {
    /// Parse one `metric<op>value` (whitespace already stripped by the
    /// caller). Operators: `<`, `<=`, `>`, `>=`, `==`.
    fn parse(expr: &str) -> Result<Atom, String> {
        let expr = expr.trim();
        // Scan left-to-right for the *first* comparison operator, preferring
        // the two-char form at each position so `<=` isn't truncated to `<`
        // (and so we split on the leftmost op, not whichever the array lists
        // first).
        let mut found: Option<(&str, usize, usize)> = None;
        for (i, _) in expr.char_indices() {
            let rest = &expr[i..];
            if rest.starts_with("<=") || rest.starts_with(">=") || rest.starts_with("==") {
                found = Some((&expr[i..i + 2], i, 2));
                break;
            }
            if rest.starts_with('<') || rest.starts_with('>') {
                found = Some((&expr[i..i + 1], i, 1));
                break;
            }
        }
        let (op, pos, oplen) = found
            .ok_or_else(|| format!("no comparison operator in `{expr}` (use < <= > >= ==)"))?;
        let metric_s = expr[..pos].trim();
        let rhs_s = expr[pos + oplen..].trim();
        let metric = Metric::parse(metric_s)
            .ok_or_else(|| format!("unknown metric `{metric_s}` (one of {METRICS:?})"))?;
        let rhs = rhs_s
            .parse::<f64>()
            .map_err(|_| format!("bad threshold `{rhs_s}`"))?;
        let cmp = match op {
            "<" => CmpOp::Lt,
            "<=" => CmpOp::Le,
            ">" => CmpOp::Gt,
            ">=" => CmpOp::Ge,
            "==" => CmpOp::Eq,
            _ => unreachable!(),
        };
        Ok(Atom {
            metric,
            op: cmp,
            rhs,
        })
    }

    fn holds(&self, ctx: &dyn DebugState) -> bool {
        self.op.eval(self.metric.eval(ctx), self.rhs)
    }
}

/// Boolean join between atoms (#72 §4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Join {
    And,
    Or,
}

/// A conditional breakpoint: one or more [`Atom`]s joined by `&&`/`||`,
/// evaluated strictly left-to-right (no operator precedence — matches the
/// issue's minimal-viable spec; parentheses are stripped). Pause when the
/// chain evaluates true.
#[derive(Clone, Debug)]
struct Condition {
    first: Atom,
    rest: Vec<(Join, Atom)>,
    /// Normalized source text, for display / dedup.
    raw: String,
}

impl Condition {
    fn parse(expr: &str) -> Result<Condition, String> {
        // Parentheses are advisory only (no precedence), so drop them.
        let cleaned: String = expr.chars().filter(|c| !matches!(c, '(' | ')')).collect();
        // Split into atoms, remembering the joiner before each.
        let mut atoms: Vec<(Option<Join>, &str)> = Vec::new();
        let bytes = cleaned.as_bytes();
        let mut start = 0usize;
        let mut i = 0usize;
        let mut pending: Option<Join> = None;
        while i + 1 < bytes.len() {
            let two = &cleaned[i..i + 2];
            let join = match two {
                "&&" => Some(Join::And),
                "||" => Some(Join::Or),
                _ => None,
            };
            if let Some(j) = join {
                atoms.push((pending, &cleaned[start..i]));
                pending = Some(j);
                i += 2;
                start = i;
            } else {
                i += 1;
            }
        }
        atoms.push((pending, &cleaned[start..]));

        let mut iter = atoms.into_iter();
        let Some((_, first_s)) = iter.next() else {
            return Err("empty condition".into());
        };
        let first = Atom::parse(first_s)?;
        let mut rest = Vec::new();
        for (join, s) in iter {
            let join = join.ok_or("malformed compound condition (dangling &&/||)")?;
            rest.push((join, Atom::parse(s)?));
        }
        // The cleaned source (whitespace/parens removed) is the display form.
        Ok(Condition {
            first,
            rest,
            raw: cleaned,
        })
    }

    fn holds(&self, ctx: &dyn DebugState) -> bool {
        let mut acc = self.first.holds(ctx);
        for (join, atom) in &self.rest {
            let v = atom.holds(ctx);
            acc = match join {
                Join::And => acc && v,
                Join::Or => acc || v,
            };
        }
        acc
    }
}

/// Context-sensitive completion candidates for the REPL line editor (and
/// the `complete` command). `before` is the line text up to the start of
/// the word being completed; `word` is that partial word. Pure so it can
/// be unit-tested without a terminal.
/// Filesystem completions for a path argument (`save`/`load`/`sweep`/
/// `source`). `word` is the whole path token typed so far; the returned
/// candidates carry its directory prefix (so they replace the token whole),
/// directories get a trailing `/`, and dotfiles are hidden unless the
/// prefix opens with a dot.
fn path_candidates(word: &str) -> Vec<String> {
    // Split into the directory to list and the basename prefix to match.
    let (dir, prefix) = match word.rfind('/') {
        Some(i) => (&word[..=i], &word[i + 1..]), // dir keeps its trailing '/'
        None => ("", word),
    };
    let read_from = if dir.is_empty() { "." } else { dir };
    let Ok(entries) = std::fs::read_dir(read_from) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with(prefix) {
            continue;
        }
        if name.starts_with('.') && !prefix.starts_with('.') {
            continue;
        }
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let mut cand = format!("{dir}{name}");
        if is_dir {
            cand.push('/');
        }
        out.push(cand);
    }
    out.sort();
    out
}

fn completion_candidates(reg: Option<&RegisteredOptions>, before: &str, word: &str) -> Vec<String> {
    let toks: Vec<&str> = before.split_whitespace().collect();
    let starts = |opts: &[&str]| -> Vec<String> {
        opts.iter()
            .filter(|c| c.starts_with(word))
            .map(|c| c.to_string())
            .collect()
    };
    let opt_names = || -> Vec<String> {
        reg.map(|r| {
            r.registered_options_in_order()
                .iter()
                .map(|o| o.name.clone())
                .filter(|n| n.starts_with(word))
                .collect()
        })
        .unwrap_or_default()
    };
    match toks.as_slice() {
        [] => starts(COMMANDS),
        ["set"] => {
            let mut v = starts(&["mu", "opt"]);
            v.extend(starts(&BLOCK_NAMES));
            v
        }
        ["set", "opt"] | ["get", "opt"] | ["get"] | ["opt"] | ["options"] => opt_names(),
        // After `set opt <name>`, complete the option's valid values.
        ["set", "opt", name] => reg
            .and_then(|r| r.get_option(name))
            .map(|o| {
                o.valid_strings
                    .iter()
                    .map(|e| e.value.clone())
                    .filter(|v| v.starts_with(word) && v != "*")
                    .collect()
            })
            .unwrap_or_default(),
        ["stop-at"] | ["stopat"] => starts(CHECKPOINTS),
        ["break", "if"] | ["b", "if"] => starts(METRICS),
        ["break", "on"] | ["b", "on"] => starts(EVENTS),
        ["break"] | ["b"] => starts(&["if", "on", "clear", "del"]),
        ["watchpoint"] | ["wp"] => starts(&BLOCK_NAMES),
        ["print"] | ["p"] | ["watch"] | ["display"] => {
            let mut v = starts(&BLOCK_NAMES);
            v.extend(starts(&[
                "mu",
                "obj",
                "inf_pr",
                "inf_du",
                "err",
                "compl",
                "iter",
                "kkt",
                "active",
                "inactive",
                "residuals",
                "equation",
                "rank",
            ]));
            v
        }
        ["viz"] | ["plot"] => {
            let mut v = starts(&BLOCK_NAMES);
            v.extend(starts(&["kkt", "L"]));
            v
        }
        ["complete"] => starts(COMMANDS),
        // Path arguments: complete against the filesystem.
        ["save"] | ["load"] | ["sweep"] | ["source"] => path_candidates(word),
        // `load <file> [block]` — the optional second arg names a block.
        ["load", _] => starts(&BLOCK_NAMES),
        _ => Vec::new(),
    }
}

/// rustyline helper: supplies Tab completion against the live command /
/// option vocabulary. Hinting / highlighting / validation are the
/// no-op derived defaults.
#[derive(Helper, Hinter, Highlighter, Validator)]
struct DbgHelper {
    reg: Option<Rc<RegisteredOptions>>,
}

impl Completer for DbgHelper {
    type Candidate = Pair;
    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let before = &line[..pos];
        let start = before
            .rfind(char::is_whitespace)
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &before[start..];
        let cands = completion_candidates(self.reg.as_deref(), &before[..start], word);
        let pairs = cands
            .into_iter()
            .map(|c| Pair {
                display: c.clone(),
                replacement: c,
            })
            .collect();
        Ok((start, pairs))
    }
}

/// Rendered constraint equations from the source model, indexed in
/// original `.nl` row order. Lets the debugger answer
/// `print equation <name|row>` with the actual algebra — the source
/// expression for a constraint, resolved by its model name. This closes
/// the loop on the residual-name labeling (`print residuals`): once a
/// culprit equation is named, the user can read it. Naming and printing
/// culprit equations rather than bare indices is the diagnostic
/// recommendation of Lee et al. (2024,
/// <https://doi.org/10.69997/sct.147875>).
pub struct EquationBook {
    /// Constraint names in original `.nl` row order (empty `String` when a
    /// row has no name, e.g. no `.row` auxfile was emitted).
    names: Vec<String>,
    /// Rendered equation text, parallel to `names`.
    equations: Vec<String>,
}

impl EquationBook {
    /// Build from parallel name / rendered-equation vectors (original
    /// `.nl` row order). Lengths are zipped to the shorter of the two.
    pub fn new(names: Vec<String>, equations: Vec<String>) -> Self {
        Self { names, equations }
    }

    /// Number of constraints with a rendered equation.
    pub fn len(&self) -> usize {
        self.equations.len()
    }

    /// True when there are no equations.
    pub fn is_empty(&self) -> bool {
        self.equations.is_empty()
    }

    /// Human label for row `i`: its model name if present, else `c[i]`
    /// (original `.nl` row index).
    fn label(&self, i: usize) -> String {
        match self.names.get(i) {
            Some(n) if !n.is_empty() => n.clone(),
            _ => format!("c[{i}]"),
        }
    }

    /// Resolve a user key to an original row index: an exact name match
    /// first, else the key parsed as a `usize` row index.
    fn resolve(&self, key: &str) -> Option<usize> {
        if let Some(i) = self.names.iter().position(|n| n == key) {
            return Some(i);
        }
        key.parse::<usize>()
            .ok()
            .filter(|&i| i < self.equations.len())
    }
}

/// Maximum number of named culprits listed inline in a structural
/// finding before it switches to a "+N more" tail. Keeps a pathological
/// model (hundreds of redundant rows) from flooding the report while
/// still reporting the full count — no silent truncation.
const MAX_STRUCT_NAMES: usize = 10;

/// Maximum singular values echoed inline by `print rank` before the tail
/// is elided (the full spectrum is always in the JSON payload).
const MAX_SINGULAR_VALUES_SHOWN: usize = 16;

/// Maximum implicated rows listed inline by `print rank` before a
/// "+N more" tail. Same no-silent-truncation rule as [`MAX_STRUCT_NAMES`].
const MAX_RANK_CULPRITS: usize = 12;

/// Structural rank analysis of the *equality* constraint Jacobian,
/// after the Dulmage–Mendelsohn decomposition used by IDAES's
/// `DiagnosticsToolbox`. The Hessian-free, iterate-independent sparsity
/// pattern alone tells us whether a subset of equations is
/// over-determined — more equations than the variables they jointly
/// touch — which forces at least one of them to be redundant or
/// mutually inconsistent (a structurally singular Jacobian, LICQ
/// failure).
///
/// The payoff is *naming* those rows. The solver's δ_c dual
/// regularization and wrong-inertia flags detect rank deficiency but
/// report it as a scalar; this book maps the dependent rows back to the
/// model's equation names so `diagnose` can say `mass_balance` instead
/// of "equation 13". Tracing a singular system to *named* equations is
/// exactly the roadblock Lee et al. (2024) identify for
/// equation-oriented model debugging. See
/// <https://doi.org/10.69997/sct.147875>.
pub struct StructureBook {
    /// Equality-row × variable incidence graph (built from the source
    /// model's Jacobian sparsity).
    inc: EqualityIncidence,
    /// Constraint names in original `.nl` row order (empty `String`
    /// when a row has no name).
    con_names: Vec<String>,
    /// Variable names in original column order (empty `String` when a
    /// column has no name).
    var_names: Vec<String>,
}

impl StructureBook {
    /// Build from the equality incidence graph plus the model's
    /// constraint and variable name vectors (original order). The
    /// incidence rows index into `con_names` via
    /// `inc.eq_row_inner_idx`; the incidence columns index `var_names`
    /// directly.
    pub fn new(inc: EqualityIncidence, con_names: Vec<String>, var_names: Vec<String>) -> Self {
        Self {
            inc,
            con_names,
            var_names,
        }
    }

    /// Label for equality-incidence row `eq_row`: the source model's
    /// constraint name if present, else `c[<orig row>]`.
    fn con_label(&self, eq_row: usize) -> String {
        let orig = self.inc.eq_row_inner_idx[eq_row];
        match self.con_names.get(orig) {
            Some(n) if !n.is_empty() => n.clone(),
            _ => format!("c[{orig}]"),
        }
    }

    /// Label for variable column `v`: the source model's variable name
    /// if present, else `x[v]`.
    fn var_label(&self, v: usize) -> String {
        match self.var_names.get(v) {
            Some(n) if !n.is_empty() => n.clone(),
            _ => format!("x[{v}]"),
        }
    }

    /// Join up to [`MAX_STRUCT_NAMES`] labels, appending an explicit
    /// "+N more" tail when truncated so nothing is dropped silently.
    fn join_capped(labels: &[String]) -> String {
        if labels.len() <= MAX_STRUCT_NAMES {
            labels.join(", ")
        } else {
            let head = labels[..MAX_STRUCT_NAMES].join(", ");
            let more = labels.len() - MAX_STRUCT_NAMES;
            format!("{head}, … (+{more} more)")
        }
    }

    /// Run the structural pass and return `diagnose` findings.
    ///
    /// Only the *over-determined* block is reported: it names the
    /// candidate dependent (redundant / inconsistent) equations behind
    /// a singular Jacobian. The under-determined block is deliberately
    /// suppressed — an NLP with more variables than equality
    /// constraints is the normal, well-posed case (the remaining
    /// degrees of freedom are pinned by the objective, bounds, and
    /// inequalities), so flagging it would fire on nearly every model.
    fn findings(&self) -> Vec<(&'static str, &'static str, String)> {
        let mut out = Vec::new();
        if self.inc.n_eq_rows() == 0 {
            return out;
        }
        let matching = hopcroft_karp(&self.inc);
        let dm = DulmageMendelsohnPartition::from_matching(&self.inc, &matching);
        if dm.over_rows.is_empty() {
            return out;
        }

        // over_rows.len() == over_cols.len() + (unmatched rows); the
        // unmatched count is the minimum number of structurally
        // redundant equations.
        let excess = dm.over_rows.len().saturating_sub(dm.over_cols.len());
        let eq_labels: Vec<String> = dm.over_rows.iter().map(|&r| self.con_label(r)).collect();
        let var_labels: Vec<String> = dm.over_cols.iter().map(|&v| self.var_label(v)).collect();
        let eqs = Self::join_capped(&eq_labels);
        let shared = if var_labels.is_empty() {
            "no variables".to_string()
        } else {
            Self::join_capped(&var_labels)
        };
        out.push((
            "warning",
            "structural_singularity",
            format!(
                "Constraint Jacobian is structurally singular (Dulmage–Mendelsohn): {} equation(s) \
                 over-determine the {} variable(s) they jointly touch ({}), so ≥{} of them must be \
                 redundant or mutually inconsistent (LICQ fails on this block). Candidate \
                 dependent equations: {}. Inspect them with `print equation <name>`; this names \
                 the rows behind any δ_c dual-regularization / wrong-inertia signal.",
                dm.over_rows.len(),
                dm.over_cols.len(),
                shared,
                excess.max(1),
                eqs
            ),
        ));
        out
    }
}

pub struct SolverDebugger {
    mode: DebugMode,
    reg: Option<Rc<RegisteredOptions>>,
    /// Pause at the next checkpoint (one-shot, re-armed by `step`).
    step: bool,
    /// Pause once iteration ≥ this value.
    run_to: Option<i32>,
    /// Iterations to break at.
    breaks: Vec<i32>,
    /// One-shot iteration breakpoints (`tbreak`), removed when hit.
    temp_breaks: Vec<i32>,
    /// Command lists attached to iteration breakpoints (`commands N …`):
    /// run automatically when iteration N is paused at.
    bp_commands: HashMap<i32, Vec<String>>,
    /// Conditional breakpoints (`break if mu<1e-4`): pause when any holds.
    conds: Vec<Condition>,
    /// Data watchpoints (`watchpoint x[3]`): pause when a value changes.
    watchpoints: Vec<WatchPoint>,
    /// μ-stall tracking for the `mu_stalled` event.
    last_mu: Option<f64>,
    mu_stall: u32,
    /// True while between `pre_restoration_entry` and
    /// `post_restoration_exit` — marks pauses fired by the inner IPM.
    in_restoration: bool,
    /// Once true, never pause again (`detach`).
    detached: bool,
    /// Whether the JSON `hello` handshake has been emitted (once per
    /// session, at the first checkpoint).
    hello_sent: bool,
    /// Pause at iteration checkpoints (false for `--debug-on-error`,
    /// which runs freely until the terminal checkpoint).
    pause_iters: bool,
    /// Pause at the terminal (post-mortem) checkpoint.
    pause_terminal: bool,
    /// At the terminal checkpoint, pause only when the solve failed.
    terminal_only_on_error: bool,
    /// Honor a pending SIGINT (Ctrl-C) by pausing at the next iteration.
    interruptible: bool,
    /// Emit a per-iteration `progress` event (JSON mode) when running
    /// between pauses, so a visual debugger can show live progress.
    emit_progress: bool,
    /// One-shot: pause at the very next checkpoint of *any* kind (set by
    /// `stepi`, for walking through sub-iteration phases).
    sub_step: bool,
    /// Checkpoint kinds (by name) to always pause at (`stop-at`).
    stop_at: HashSet<&'static str>,
    /// Events to break on (`break on <event>`), from [`EVENTS`].
    break_events: HashSet<&'static str>,
    /// Per-iteration primal-dual snapshots for `goto`/`restart`, keyed by
    /// iteration index. Capped at [`SNAPSHOT_CAP`] (oldest evicted).
    snapshots: BTreeMap<i32, Box<dyn pounce_common::debug::IterSnapshot>>,
    /// Shared slot for `resolve` to request a fresh solve from the
    /// current point with staged options. `None` disables `resolve`.
    restart: Option<RestartCell>,
    /// rustyline editor for the human REPL on a TTY (history + Tab +
    /// Ctrl-R). `None` for JSON mode or when stdin isn't a terminal, in
    /// which case a plain line reader is used.
    editor: Option<Editor<DbgHelper, FileHistory>>,
    /// Where REPL history is persisted, if a home directory was found.
    hist_path: Option<PathBuf>,
    /// Background stdin reader (JSON mode) enabling async `{"cmd":"pause"}`
    /// during a run. `None` in REPL mode.
    pump: Option<StdinPump>,
    /// Expressions to auto-print at every pause (`watch`). Each is a
    /// `print` target (block, `dx`, scalar, `kkt`).
    watches: Vec<String>,
    /// A debugger script (file path) to run once at the first pause
    /// (`--debug-script`); consumed on use.
    pending_script: Option<String>,
    /// Option edits accepted at the prompt. Validated against the
    /// registry; surfaced to the caller after the solve. Not applied to
    /// already-built strategies mid-solve (see `staged_options`).
    staged: Vec<(String, String)>,
    /// Active `sweep` / `multistart` run, if any. Driven at the terminal
    /// checkpoint across re-solves (see [`SolverDebugger::drive_sweep`]).
    sweep: Option<SweepState>,
    /// Consecutive Ctrl-C presses at the REPL prompt with no command in
    /// between. The first cancels the line (readline convention); a second
    /// quits the solve — a discoverable Ctrl-C escape hatch that mirrors the
    /// running-mode double-tap. Reset whenever a real line is entered.
    prompt_interrupts: u8,
    /// Rendered constraint equations from the source model (`.nl`), for the
    /// `print equation <name|row>` command. `None` when no model was wired in
    /// (e.g. a non-`.nl` entry point). See Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>) on naming culprit equations.
    equation_book: Option<EquationBook>,
    /// Structural rank analysis of the source model's equality Jacobian,
    /// for the `diagnose` command's `structural_singularity` finding.
    /// `None` when no `.nl` model was wired in. See Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>).
    structure_book: Option<StructureBook>,
}

impl SolverDebugger {
    /// Fully interactive: pause at the first iteration and at the
    /// terminal checkpoint.
    pub fn new(mode: DebugMode, reg: Option<Rc<RegisteredOptions>>) -> Self {
        Self {
            mode,
            reg,
            // Pause at the very first checkpoint so the user has control
            // before iteration 0's step is computed.
            step: true,
            run_to: None,
            breaks: Vec::new(),
            temp_breaks: Vec::new(),
            bp_commands: HashMap::new(),
            conds: Vec::new(),
            watchpoints: Vec::new(),
            last_mu: None,
            mu_stall: 0,
            in_restoration: false,
            detached: false,
            hello_sent: false,
            pause_iters: true,
            pause_terminal: true,
            terminal_only_on_error: false,
            interruptible: true,
            emit_progress: true,
            sub_step: false,
            stop_at: HashSet::new(),
            break_events: HashSet::new(),
            snapshots: BTreeMap::new(),
            restart: None,
            editor: None,
            hist_path: None,
            pump: None,
            watches: Vec::new(),
            pending_script: None,
            staged: Vec::new(),
            sweep: None,
            prompt_interrupts: 0,
            equation_book: None,
            structure_book: None,
        }
    }

    /// A debugger that stays **quiet** (never pauses) until [`arm`]ed. Used as
    /// the on-demand sub-solve hook for the branch-and-bound tree debugger:
    /// it sees a node's relaxation solve only when the user steps into it.
    ///
    /// [`arm`]: DebugHook::arm
    pub fn quiet(mode: DebugMode, reg: Option<Rc<RegisteredOptions>>) -> Self {
        let mut d = Self::new(mode, reg);
        d.step = false;
        d.pause_iters = false;
        d.pause_terminal = false;
        d.detached = true;
        d
    }

    /// Queue a debugger script to run once at the first pause.
    pub fn with_script(mut self, path: String) -> Self {
        self.pending_script = Some(path);
        self
    }

    /// Attach the source model's rendered constraint equations, enabling
    /// `print equation <name|row>`. Wired in on the `.nl` entry path
    /// (see Lee et al. 2024, <https://doi.org/10.69997/sct.147875>).
    pub fn set_equation_book(&mut self, book: EquationBook) {
        self.equation_book = Some(book);
    }

    /// Attach the source model's structural rank analysis, enabling the
    /// `diagnose` command's `structural_singularity` finding (named
    /// dependent equations). Wired in on the `.nl` entry path alongside
    /// the equation book. See Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>).
    pub fn set_structure_book(&mut self, book: StructureBook) {
        self.structure_book = Some(book);
    }

    /// Enable the `resolve` command, wiring the shared restart slot the
    /// CLI's re-solve loop reads.
    pub fn with_restart(mut self, cell: RestartCell) -> Self {
        self.restart = Some(cell);
        self
    }

    /// Post-mortem: run freely, then drop in at the terminal checkpoint
    /// only if the solve did not succeed (`--debug-on-error`).
    pub fn on_error(mode: DebugMode, reg: Option<Rc<RegisteredOptions>>) -> Self {
        Self {
            step: false,
            pause_iters: false,
            terminal_only_on_error: true,
            ..Self::new(mode, reg)
        }
    }

    /// Attach-on-demand: run normally and only drop in when the user
    /// presses Ctrl-C (`--debug-on-interrupt`). No automatic iter or
    /// terminal pauses.
    pub fn on_interrupt(mode: DebugMode, reg: Option<Rc<RegisteredOptions>>) -> Self {
        Self {
            step: false,
            pause_iters: false,
            pause_terminal: false,
            ..Self::new(mode, reg)
        }
    }

    /// Option edits accepted at the prompt (validated). The caller may
    /// re-run the solve with these applied.
    pub fn staged_options(&self) -> &[(String, String)] {
        &self.staged
    }

    fn should_pause(&mut self, iter: i32) -> bool {
        if self.detached {
            return false;
        }
        if self.step {
            return true;
        }
        if let Some(t) = self.run_to {
            if iter >= t {
                self.run_to = None;
                return true;
            }
        }
        if self.breaks.contains(&iter) {
            return true;
        }
        // One-shot breakpoints fire once then delete themselves.
        if let Some(pos) = self.temp_breaks.iter().position(|&b| b == iter) {
            self.temp_breaks.remove(pos);
            return true;
        }
        false
    }

    /// First conditional breakpoint that holds at the current state, if
    /// any. Returns its source text (for the pause banner / event).
    fn matched_condition(&self, ctx: &dyn DebugState) -> Option<String> {
        if self.detached {
            return None;
        }
        self.conds
            .iter()
            .find(|c| c.holds(ctx))
            .map(|c| c.raw.clone())
    }

    /// First armed event that fires at the current checkpoint/state, if
    /// any. Events are derived from observable state, so they're evaluated
    /// at the checkpoint where the relevant quantity is meaningful.
    fn matched_event(&self, ctx: &dyn DebugState) -> Option<&'static str> {
        if self.detached || self.break_events.is_empty() {
            return None;
        }
        let cp = ctx.checkpoint();
        // Tiny-step threshold mirrors the solver's own scale.
        let tiny = 1e-10;
        EVENTS.iter().copied().find(|&e| {
            self.break_events.contains(e)
                && match e {
                    "resto_entered" => cp == Checkpoint::PreRestoration,
                    "resto_exited" => cp == Checkpoint::PostRestoration,
                    "regularized" => {
                        cp == Checkpoint::AfterSearchDirection && ctx.regularization() > 0.0
                    }
                    "tiny_step" => {
                        cp == Checkpoint::AfterSearchDirection
                            && ctx
                                .delta_block("x")
                                .map(|v| v.iter().fold(0.0_f64, |m, &x| m.max(x.abs())) < tiny)
                                .unwrap_or(false)
                    }
                    "ls_rejected" => cp == Checkpoint::AfterStep && ctx.ls_count() > 1,
                    "mu_stalled" => cp == Checkpoint::IterStart && self.mu_stall >= MU_STALL_ITERS,
                    "nan" => !ctx.nlp_error().is_finite() || !ctx.objective().is_finite(),
                    _ => false,
                }
        })
    }

    /// Update μ-stall tracking once per iteration (drives `mu_stalled`).
    fn update_mu_stall(&mut self, mu: f64) {
        if let Some(last) = self.last_mu {
            if (mu - last).abs() <= 1e-12 * last.abs().max(1.0) {
                self.mu_stall += 1;
            } else {
                self.mu_stall = 0;
            }
        }
        self.last_mu = Some(mu);
    }

    /// First watchpoint whose value changed (beyond its threshold) since
    /// the previous iteration. Updates the stored baselines.
    fn matched_watchpoint(&mut self, ctx: &dyn DebugState) -> Option<String> {
        if self.detached {
            return None;
        }
        let mut hit = None;
        for wp in self.watchpoints.iter_mut() {
            let Some(full) = ctx.block(&wp.block) else {
                continue;
            };
            let cur: Vec<f64> = match wp.idx {
                Some(i) => match full.get(i) {
                    Some(&v) => vec![v],
                    None => continue,
                },
                None => full,
            };
            if let Some(prev) = &wp.last {
                if prev.len() == cur.len() {
                    let changed = prev
                        .iter()
                        .zip(&cur)
                        .any(|(p, c)| (p - c).abs() > wp.threshold);
                    if changed && hit.is_none() {
                        hit = Some(wp.raw.clone());
                    }
                }
            }
            wp.last = Some(cur);
        }
        hit
    }

    // ---- command engine -----------------------------------------------

    fn dispatch(&mut self, line: &str, ctx: &mut dyn DebugState) -> CmdOut {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let Some(&verb) = toks.first() else {
            return CmdOut::ok(vec![]); // empty line: reprompt
        };
        let rest = &toks[1..];
        match verb {
            "help" | "h" | "?" => self.cmd_help(),
            "info" | "i" => self.cmd_info(ctx),
            "print" | "p" => self.cmd_print(rest, ctx),
            // `step` → next iter_start; `step sub` (or `stepi`/`si`) →
            // next checkpoint of any kind (issue #72's step ["sub"]).
            "step" | "s" | "n" | "next" if rest.first() == Some(&"sub") => {
                self.sub_step = true;
                CmdOut::ok(vec![
                    "stepping to the next checkpoint (sub-iteration)".into()
                ])
                .flow(Flow::Resume)
            }
            "step" | "s" | "n" | "next" => {
                self.step = true;
                CmdOut::ok(vec!["stepping one iteration".into()]).flow(Flow::Resume)
            }
            "stepi" | "si" => {
                self.sub_step = true;
                CmdOut::ok(vec![
                    "stepping to the next checkpoint (sub-iteration)".into()
                ])
                .flow(Flow::Resume)
            }
            "continue" | "c" | "cont" => {
                self.step = false;
                self.sub_step = false;
                self.run_to = None;
                CmdOut::ok(vec!["continuing".into()]).flow(Flow::Resume)
            }
            "run" | "r" => self.cmd_run(rest),
            "break" | "b" => self.cmd_break(rest),
            "tbreak" | "tb" => match rest.first().and_then(|s| s.parse::<i32>().ok()) {
                Some(n) => {
                    if !self.temp_breaks.contains(&n) {
                        self.temp_breaks.push(n);
                    }
                    CmdOut::ok(vec![format!("temporary breakpoint at iteration {n}")])
                }
                None => CmdOut::err("usage: tbreak <iteration>"),
            },
            "watchpoint" | "wp" => self.cmd_watchpoint(rest, ctx),
            "commands" => self.cmd_commands(rest),
            "stop-at" | "stopat" => self.cmd_stop_at(rest),
            "progress" => match rest.first().copied() {
                Some("on") | None => {
                    self.emit_progress = true;
                    CmdOut::ok(vec!["progress events on".into()])
                }
                Some("off") => {
                    self.emit_progress = false;
                    CmdOut::ok(vec!["progress events off".into()])
                }
                _ => CmdOut::err("usage: progress [on|off]"),
            },
            "set" => self.cmd_set(rest, ctx),
            "get" => self.cmd_get(rest),
            "opt" | "options" => self.cmd_opt(rest),
            "complete" => self.cmd_complete(rest),
            "viz" | "plot" => self.cmd_viz(rest, ctx),
            "save" => self.cmd_save(rest, ctx),
            "load" => match as_nlp_mut(ctx) {
                Some(c) => self.cmd_load(rest, c),
                None => nlp_only("load"),
            },
            "sweep" => match as_nlp_mut(ctx) {
                Some(c) => self.cmd_sweep(rest, c),
                None => nlp_only("sweep"),
            },
            "multistart" => match as_nlp_mut(ctx) {
                Some(c) => self.cmd_multistart(rest, c),
                None => nlp_only("multistart"),
            },
            "goto" | "jump" => self.cmd_goto(rest, ctx),
            "restart" => match self.snapshots.keys().next().copied() {
                Some(k) => self.restore_to(k, ctx),
                None => CmdOut::err("no snapshots captured yet"),
            },
            "resolve" | "re-solve" => match as_nlp(ctx) {
                Some(c) => self.cmd_resolve(c),
                None => nlp_only("resolve"),
            },
            "ask" | "explain" | "claude" => self.cmd_ask(rest, ctx),
            "watch" | "display" => self.cmd_watch(rest),
            "diff" => self.cmd_diff(ctx),
            "diagnose" | "diag" => match as_nlp(ctx) {
                Some(c) => self.cmd_diagnose(c),
                None => nlp_only("diagnose"),
            },
            "source" => self.cmd_source(rest, ctx),
            "detach" => {
                self.detached = true;
                self.step = false;
                self.run_to = None;
                CmdOut::ok(vec!["detached — solving to completion".into()]).flow(Flow::Resume)
            }
            // A `pause` received while already paused is a no-op; the
            // meaningful use is async, consumed mid-run by `try_take_pause`.
            "pause" => CmdOut::ok(vec!["already paused".into()]),
            // Easter egg — not in COMMANDS / help / Tab, so it stays hidden.
            "coffee" | "brew" | "espresso" => self.cmd_coffee(),
            "quit" | "q" | "exit" => CmdOut::ok(vec!["stopping solve".into()]).flow(Flow::Stop),
            other => CmdOut::err(format!("unknown command `{other}` (try `help`)")),
        }
    }

    /// `coffee` — a hidden treat. Prints a steaming mug in colour (TTY +
    /// `NO_COLOR`-respecting, like the banner). Pure output, no solver
    /// effect; every IPM deserves a coffee break.
    fn cmd_coffee(&self) -> CmdOut {
        let color = matches!(self.mode, DebugMode::Repl)
            && std::io::stderr().is_terminal()
            && std::env::var_os("NO_COLOR").is_none();
        let paint = |r: u8, g: u8, b: u8, s: &str| -> String {
            if color {
                format!("\x1b[38;2;{r};{g};{b}m{s}\x1b[0m")
            } else {
                s.to_string()
            }
        };
        // Palette: ceramic white, dark-roast & medium brown, gray steam.
        let cup = |s: &str| paint(0xEC, 0xEC, 0xEF, s);
        let dark = |s: &str| paint(0x5A, 0x32, 0x1E, s);
        let brew = |s: &str| paint(0x96, 0x5F, 0x37, s);
        let steam = |s: &str| paint(0xB4, 0xB9, 0xC3, s);
        let lines = vec![
            String::new(),
            format!("     {}", steam(") )  )")),
            format!("    {}", steam("( (  (")),
            format!("   {}", cup("._________.")),
            format!("   {}{}{}", cup("|"), dark("~~~~~~~~"), cup("|_")),
            format!("   {}{}{}", cup("|  "), brew("COFFEE"), cup("| |")),
            format!("   {}{}{}", cup("|  "), dark("~~~~~~"), cup("| |")),
            format!("   {}", cup("|________|_|")),
            format!("    {}", cup("\\________/")),
            format!("      {}", brew("a fresh cup for a stuck solve")),
            String::new(),
        ];
        CmdOut::ok(lines).with_data(serde_json::json!({"easter_egg": "coffee"}))
    }

    fn cmd_help(&self) -> CmdOut {
        let lines = vec![
            "commands:".into(),
            "  info | i                 summary of the current iterate".into(),
            "  print | p <what>         x|s|y_c|y_d|z_l|z_u|v_l|v_u | dx (step) |".into(),
            "                           mu|obj|inf_pr|inf_du|err|compl|iter | kkt | active | inactive".into(),
            "  print residuals [pr|du] [k]  top-k largest-magnitude residuals (default k=10)".into(),
            "  print equation [name|row]    source algebra of a constraint, by model name or row".into(),
            "  print rank                   SVD rank of the equality Jacobian; names dependent equations".into(),
            "  step | s | n             run one iteration, pause again".into(),
            "  stepi | si | step sub    run to the next checkpoint (into sub-iteration phases)".into(),
            "  progress [on|off]        toggle per-iteration progress events (JSON mode)".into(),
            "  stop-at <cp>             always pause at a checkpoint: after_mu|after_search_dir|after_step".into(),
            "  continue | c             run to the next breakpoint".into(),
            "  run | r <N>              run until iteration N".into(),
            "  break | b [N|clear|del N] set/list/clear breakpoints".into(),
            "  break if <m><op><v>      conditional bp; m in mu|inf_pr|inf_du|obj|err|iter,".into(),
            "                           op in < <= > >= ==  (e.g. break if inf_pr<1e-6)".into(),
            "  break on <event>         event bp: resto_entered|resto_exited|regularized|".into(),
            "                           tiny_step|ls_rejected|mu_stalled|nan".into(),
            "  tbreak <N>               one-shot breakpoint (deletes after firing)".into(),
            "  watchpoint <blk>[<i>] [τ] pause when a value changes by > τ (alias wp)".into(),
            "  commands <N> <c>;<c>…    auto-run commands when iter N's breakpoint hits".into(),
            "  set mu <v>               overwrite the barrier parameter".into(),
            "  set <blk>[<i>] <v>       overwrite one component (e.g. set x[2] 1.5)".into(),
            "  set <blk> <v0,v1,...>    overwrite a whole block".into(),
            "  set opt <name> <value>   stage a solver option (validated)".into(),
            "  get opt <name>           show an option's effective value (staged or default)".into(),
            "  opt [filter]             list solver options (name/type/default)".into(),
            "  complete <prefix>        completion candidates (commands + options)".into(),
            "  viz <x|s|dx|...|kkt|L>   open the artifact in an external viewer".into(),
            "  save [path]              write the current iterate + residuals to JSON".into(),
            "  load <file> [block]      read a block (default x) from a save artifact / numeric file".into(),
            "  sweep <file>             one solve per start in <file>; tabulate outcomes".into(),
            "  multistart <N> [rel]     N restarts (uniform in each finite box; jitter else)".into(),
            "  goto <k> | restart       rewind to a captured iteration (primal-dual only)".into(),
            "  resolve                  re-solve from the current x with staged `set opt`s".into(),
            "  ask [question]           ask Claude Code (claude -p / $POUNCE_DBG_LLM) about the state".into(),
            "  watch [target|clear|del] auto-print a `print` target at every pause".into(),
            "  diff                     what changed in the iterate since the last iteration".into(),
            "  diagnose | diag          live health report: named culprit residuals, KKT inertia, stalls".into(),
            "  source <file>            run debugger commands from a file".into(),
            "  detach                   stop pausing; solve to completion".into(),
            "  quit | q                 stop the solve now".into(),
        ];
        CmdOut::ok(lines)
    }

    fn cmd_info(&self, ctx: &dyn DebugState) -> CmdOut {
        let dims: Vec<_> = ctx.block_dims();
        let dims_json: serde_json::Map<String, serde_json::Value> = dims
            .iter()
            .map(|(n, d)| ((*n).to_string(), serde_json::json!(d)))
            .collect();
        let lines = vec![
            format!("iter      = {}", ctx.iter()),
            format!("mu        = {:.6e}", ctx.mu()),
            format!("objective = {:.8e}", ctx.objective()),
            format!("inf_pr    = {:.6e}", ctx.inf_pr()),
            format!("inf_du    = {:.6e}", ctx.inf_du()),
            format!("nlp_error = {:.6e}", ctx.nlp_error()),
            format!(
                "dims      = {}",
                dims.iter()
                    .map(|(n, d)| format!("{n}:{d}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        ];
        CmdOut::ok(lines).with_data(serde_json::json!({
            "iter": ctx.iter(),
            "mu": ctx.mu(),
            "objective": ctx.objective(),
            "inf_pr": ctx.inf_pr(),
            "inf_du": ctx.inf_du(),
            "nlp_error": ctx.nlp_error(),
            "dims": dims_json,
        }))
    }

    fn cmd_print(&self, rest: &[&str], ctx: &dyn DebugState) -> CmdOut {
        let Some(&what) = rest.first() else {
            return self.cmd_info(ctx);
        };
        if what == "kkt" {
            return self.cmd_print_kkt(ctx);
        }
        if what == "active" {
            return self.cmd_print_bounds(ctx, true);
        }
        if what == "inactive" {
            return self.cmd_print_bounds(ctx, false);
        }
        if what == "residuals" || what == "resid" {
            return self.cmd_print_residuals(&rest[1..], ctx);
        }
        if what == "equation" || what == "eqn" || what == "eq" {
            return self.cmd_print_equation(&rest[1..]);
        }
        if what == "rank" {
            return match as_nlp(ctx) {
                Some(c) => self.cmd_print_rank(c),
                None => nlp_only("print rank"),
            };
        }
        // step / delta blocks: `dx`, `ds`, ... or `delta_x`.
        let delta = what.strip_prefix("d").filter(|b| is_block(ctx, b));
        if is_block(ctx, what) {
            match ctx.block(what) {
                Some(v) => CmdOut::ok(vec![fmt_vec(what, &v)])
                    .with_data(serde_json::json!({"name": what, "values": v})),
                None => CmdOut::err(format!("no iterate yet for block `{what}`")),
            }
        } else if let Some(blk) = delta {
            match ctx.delta_block(blk) {
                Some(v) => CmdOut::ok(vec![fmt_vec(&format!("d{blk}"), &v)])
                    .with_data(serde_json::json!({"name": format!("d{blk}"), "values": v})),
                None => CmdOut::err(format!("no search direction available for `d{blk}` yet")),
            }
        } else {
            let val = match what {
                "mu" => ctx.mu(),
                "obj" | "objective" => ctx.objective(),
                "inf_pr" => ctx.inf_pr(),
                "inf_du" => ctx.inf_du(),
                "err" | "nlp_error" => ctx.nlp_error(),
                "compl" | "complementarity" => ctx.complementarity(),
                "iter" => ctx.iter() as f64,
                _ => {
                    return CmdOut::err(format!(
                        "don't know how to print `{what}` (try a block name or mu|obj|inf_pr|inf_du|err|compl|iter)"
                    ))
                }
            };
            CmdOut::ok(vec![format!("{what} = {val:.10e}")])
                .with_data(serde_json::json!({"name": what, "value": val}))
        }
    }

    /// `print active` / `print inactive` — bound-slack classification per
    /// category. `active` counts bounds the iterate is pressing on (slack
    /// below `tol`) and reports the min slack; `inactive` is the mirror —
    /// it counts the bounds with room to spare (slack ≥ `tol`) and reports
    /// the max slack, the variables furthest from their bound.
    fn cmd_print_bounds(&self, ctx: &dyn DebugState, active: bool) -> CmdOut {
        let tol = 1e-6;
        let mut lines = Vec::new();
        let mut cats = serde_json::Map::new();
        for cat in ["x_l", "x_u", "s_l", "s_u"] {
            let Some(sl) = ctx.bound_slack(cat) else {
                continue;
            };
            if sl.is_empty() {
                continue;
            }
            let n = sl.len();
            if active {
                let min = sl.iter().copied().fold(f64::INFINITY, f64::min);
                let near = sl.iter().filter(|&&s| s.abs() < tol).count();
                lines.push(format!(
                    "{cat}: {n} bound(s), {near} near-active (slack<{tol:.0e}), min slack {min:.3e}"
                ));
                cats.insert(
                    cat.to_string(),
                    serde_json::json!({"n": n, "near_active": near, "min_slack": min}),
                );
            } else {
                let max = sl.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let far = sl.iter().filter(|&&s| s.abs() >= tol).count();
                lines.push(format!(
                    "{cat}: {n} bound(s), {far} inactive (slack≥{tol:.0e}), max slack {max:.3e}"
                ));
                cats.insert(
                    cat.to_string(),
                    serde_json::json!({"n": n, "inactive": far, "max_slack": max}),
                );
            }
        }
        if lines.is_empty() {
            lines.push("no bounded variables or inequality slacks".into());
        }
        CmdOut::ok(lines).with_data(serde_json::json!({"tol": tol, "categories": cats}))
    }

    /// `print residuals [primal|dual] [k]` — the `k` largest-magnitude
    /// residuals at this step, ranked. With no filter, primal
    /// (constraint) and dual (∇L) residuals are pooled and ranked
    /// together; `primal`/`dual` restrict to one space. Default `k=10`.
    /// The top primal entry equals `inf_pr`; the top dual equals
    /// `inf_du`. Args may appear in either order.
    fn cmd_print_residuals(&self, rest: &[&str], ctx: &dyn DebugState) -> CmdOut {
        let mut k: Option<usize> = None;
        let mut filter: Option<bool> = None; // Some(true)=primal, Some(false)=dual
        for &arg in rest {
            if let Ok(n) = arg.parse::<usize>() {
                k = Some(n);
            } else {
                match arg {
                    "primal" | "pr" => filter = Some(true),
                    "dual" | "du" => filter = Some(false),
                    other => {
                        return CmdOut::err(format!(
                            "usage: print residuals [primal|dual] [k] (got `{other}`)"
                        ))
                    }
                }
            }
        }
        let k = k.unwrap_or(10);

        let mut all = Vec::new();
        if filter != Some(false) {
            let Some(primal) = ctx.constraint_residuals() else {
                return CmdOut::err("no iterate yet — residuals unavailable");
            };
            all.extend(primal);
        }
        if filter != Some(true) {
            let Some(dual) = ctx.dual_residuals() else {
                return CmdOut::err("no iterate yet — residuals unavailable");
            };
            all.extend(dual);
        }

        let total = all.len();
        let top = rank_residuals(all, k);
        if top.is_empty() {
            return CmdOut::ok(vec!["no residuals at this iterate".into()])
                .with_data(serde_json::json!({"k": k, "total": total, "top": []}));
        }

        // Model names projected into the solver's split space, when the
        // problem carries them (`.col`/`.row`, no presolve). Lets a residual
        // print as `mass_balance` rather than `c[3]` — the model-vs-index
        // gap Lee et al. (2024, <https://doi.org/10.69997/sct.147875>) flag
        // for equation-oriented debugging. `None` ⇒ index labels throughout.
        // Model names are NLP-specific (.col/.row); only the NLP debugger
        // exposes them — other solvers fall back to index labels.
        let names = ctx
            .as_any()
            .and_then(|a| a.downcast_ref::<DebugCtx>())
            .and_then(|c| c.split_names());
        let name_of = |r: &Residual| resid_name(r, &names);

        let lines = top
            .iter()
            .map(|r| {
                let label = match name_of(r) {
                    Some(name) => format!("{}[{}]", r.kind.tag(), name),
                    None => format!("{}[{}]", r.kind.tag(), r.index),
                };
                format!("{:>8} = {:+.6e}   |{:.3e}|", label, r.value, r.value.abs())
            })
            .collect();
        let data: Vec<_> = top
            .iter()
            .map(|r| {
                serde_json::json!({
                    "space": r.kind.tag(),
                    "primal": r.kind.is_primal(),
                    "index": r.index,
                    "name": name_of(r),
                    "value": r.value,
                })
            })
            .collect();
        CmdOut::ok(lines).with_data(serde_json::json!({"k": k, "total": total, "top": data}))
    }

    /// `print equation [name|row]` — the source algebra of a constraint,
    /// resolved by its model name (preferred) or original `.nl` row index.
    /// With no argument, reports how many equations are available and how
    /// to address one. This is the read-side companion to the named
    /// residual labels (`print residuals`): once a culprit constraint is
    /// named, this prints what it actually says. Naming and surfacing
    /// culprit equations rather than bare indices is the diagnostic path
    /// urged by Lee et al. (2024, <https://doi.org/10.69997/sct.147875>).
    fn cmd_print_equation(&self, rest: &[&str]) -> CmdOut {
        let Some(book) = self.equation_book.as_ref() else {
            return CmdOut::err(
                "no equation source — `print equation` needs an .nl model (none was loaded)",
            );
        };
        if book.is_empty() {
            return CmdOut::err("the model has no constraint equations to print");
        }
        let Some(&key) = rest.first() else {
            return CmdOut::ok(vec![format!(
                "{} constraint equation(s) — `print equation <name|row>` to show one",
                book.len()
            )])
            .with_data(serde_json::json!({"count": book.len()}));
        };
        let Some(i) = book.resolve(key) else {
            return CmdOut::err(format!(
                "no constraint named or indexed `{key}` (have {} equation(s); try a name or 0..{})",
                book.len(),
                book.len().saturating_sub(1)
            ));
        };
        let label = book.label(i);
        let eq = &book.equations[i];
        CmdOut::ok(vec![format!("{label}:  {eq}")]).with_data(serde_json::json!({
            "index": i,
            "name": book.names.get(i).filter(|n| !n.is_empty()),
            "equation": eq,
        }))
    }

    /// `diagnose` (`diag`) — a point-in-time health report for the
    /// *current* iterate.
    ///
    /// Where the studio `diagnose` tool runs temporal heuristics over a
    /// finished solve report, this runs **live**: it reads the current KKT
    /// inertia / regularization, the named primal & dual residuals, the
    /// iterate geometry, and the debugger's own restoration / μ-stall
    /// tracking — and names the culprit equation or variable wherever it
    /// can. Tracing a numerical symptom back to the *named* equation behind
    /// it, rather than a bare row index, is the actionable-diagnostics path
    /// of Lee et al. (2024, <https://doi.org/10.69997/sct.147875>).
    ///
    /// Each finding is `{severity, code, message}` — the same shape the
    /// report-based `diagnose` emits — so a client can treat both uniformly.
    fn cmd_diagnose(&self, ctx: &DebugCtx) -> CmdOut {
        const TOL: f64 = 1e-6;
        let names = ctx.split_names();
        // (severity, code, message). Severity ranks error > warning > info.
        let mut f: Vec<(&'static str, &'static str, String)> = Vec::new();

        // --- Primal feasibility: the worst *named* constraint residual. ---
        let inf_pr = ctx.inf_pr();
        if inf_pr > TOL {
            if let Some(resids) = ctx.constraint_residuals() {
                if let Some((label, val)) = worst_named(resids, &names) {
                    let sev = if inf_pr > 1e-2 { "error" } else { "warning" };
                    f.push((
                        sev,
                        "primal_infeasible",
                        format!(
                            "Primal infeasibility {inf_pr:.2e}; worst constraint residual is \
                         {label} = {val:+.3e}. Inspect this equation's feasibility and scaling \
                         at the current point (`print equation {label}`)."
                        ),
                    ));
                }
            }
        }

        // --- Dual stationarity: the worst *named* ∇L component. ---
        let inf_du = ctx.inf_du();
        if inf_du > TOL {
            if let Some(resids) = ctx.dual_residuals() {
                if let Some((label, val)) = worst_named(resids, &names) {
                    f.push((
                        "warning",
                        "dual_infeasible",
                        format!(
                            "Dual infeasibility {inf_du:.2e}; largest stationarity residual is \
                         {label} = {val:+.3e}."
                        ),
                    ));
                }
            }
        }

        // --- KKT structural health (only once a search dir is computed). ---
        if let Some(k) = ctx.kkt() {
            if k.provides_inertia && !k.inertia_correct {
                f.push((
                    "warning",
                    "inertia_wrong",
                    format!(
                        "KKT inertia is wrong (n-={} vs expected {}): the system was \
                     indefinite/singular and the step had to be stabilized. A persistent \
                     mismatch points at a rank-deficient Jacobian or an indefinite Hessian.",
                        k.n_neg, k.expected_neg
                    ),
                ));
            }
            if k.delta_w > 1e-4 {
                f.push((
                    "info",
                    "heavy_regularization",
                    format!(
                        "Primal regularization δ_w={:.2e} applied — the Hessian was indefinite at \
                     this step. Normal near saddle points; persistent large δ_w suggests a \
                     problematic Hessian.",
                        k.delta_w
                    ),
                ));
            }
            if k.delta_c > 0.0 {
                f.push((
                    "warning",
                    "dual_regularization",
                    format!(
                    "Dual regularization δ_c={:.2e} applied — the constraint Jacobian is (near) \
                     rank-deficient (linearly dependent or redundant equalities). Inspect the \
                     equality residuals by name (`print residuals primal`).",
                    k.delta_c
                ),
                ));
            }
        }

        // --- Structural rank: name the dependent equations (DM). ---
        // Iterate-independent; localizes the δ_c / wrong-inertia signal
        // above to the specific over-determined rows by model name.
        if let Some(book) = self.structure_book.as_ref() {
            f.extend(book.findings());
        }

        // --- Numerical rank: SVD of the equality Jacobian at this point. ---
        // The numerical complement to the structural pass above: catches
        // *value* dependencies a full sparsity pattern hides, and localizes
        // the δ_c signal to specific equations even when the structure is
        // nominally full rank. Iterate-dependent (it factors J_c at x).
        if let Some(rep) = ctx.rank_report() {
            if rep.is_rank_deficient() {
                let culprits: Vec<String> = rep
                    .culprits
                    .iter()
                    .take(MAX_RANK_CULPRITS)
                    .map(|c| rank_row_label(&rep.rows[c.row], &names))
                    .collect();
                let named = if culprits.is_empty() {
                    String::new()
                } else {
                    format!(" Implicated equations: {}.", culprits.join(", "))
                };
                f.push((
                    "warning",
                    "rank_deficient_jacobian",
                    format!(
                        "Equality Jacobian J_c is numerically rank-deficient at this iterate: \
                         rank {}/{} (deficiency {}), σ_min={:.2e}, cond={}. Linearly dependent \
                         or redundant equality constraints — the root cause behind δ_c \
                         regularization / wrong inertia.{named}",
                        rep.rank,
                        rep.n_rows(),
                        rep.deficiency(),
                        rep.sigma_min(),
                        fmt_cond(rep.cond),
                    ),
                ));
            }
        }

        // --- Multiplier magnitude: constraint-qualification / scaling. ---
        let mut max_mult = 0.0_f64;
        for blk in ["y_c", "y_d", "z_l", "z_u", "v_l", "v_u"] {
            if let Some(v) = ctx.block(blk) {
                max_mult = v.iter().fold(max_mult, |m, &x| m.max(x.abs()));
            }
        }
        if max_mult > 1e8 {
            f.push((
                "warning",
                "large_multipliers",
                format!(
                "Largest multiplier magnitude is {max_mult:.2e}. Very large multipliers signal a \
                 constraint-qualification failure or poor scaling — consider rescaling the \
                 offending rows."
            ),
            ));
        }

        // --- Iterate geometry: variable bounds pressed at this point. ---
        let mut pinned = 0usize;
        for cat in ["x_l", "x_u"] {
            if let Some(sl) = ctx.bound_slack(cat) {
                pinned += sl.iter().filter(|&&s| s.abs() < TOL).count();
            }
        }
        if pinned > 0 {
            f.push((
                "info",
                "bounds_pinned",
                format!(
                    "{pinned} variable bound(s) are active (slack < {TOL:.0e}). Active bounds are \
                 expected at a solution, but a large count early can throttle the line search."
                ),
            ));
        }

        // --- Line search / step length at this iteration. ---
        let (alpha_pr, _) = ctx.alpha();
        if ctx.iter() > 0 && alpha_pr > 0.0 && alpha_pr < 1e-6 {
            f.push((
                "warning",
                "tiny_step",
                format!(
                    "Accepted primal step α_pr={alpha_pr:.2e} is tiny — the line search is barely \
                 moving. Often a poor search direction or an ill-conditioned KKT system."
                ),
            ));
        }
        let ls = ctx.ls_count();
        if ls >= 10 {
            f.push((
                "warning",
                "heavy_line_search",
                format!(
                "Line search needed {ls} trial points for the accepted step — search-direction \
                 quality may be poor (check Hessian accuracy)."
            ),
            ));
        }

        // --- Temporal flags the debugger already tracks across iters. ---
        if self.in_restoration {
            f.push((
                "warning",
                "in_restoration",
                "Currently inside feasibility restoration: the line search could not make \
                 progress on the original problem at the working point."
                    .to_string(),
            ));
        }
        if self.mu_stall >= MU_STALL_ITERS {
            f.push((
                "warning",
                "mu_stalled",
                format!(
                    "μ has not decreased for {} consecutive iterations — the barrier is stuck. \
                 Try mu_strategy=adaptive or a smaller mu_init.",
                    self.mu_stall
                ),
            ));
        }

        // --- Healthy fallback. ---
        if f.is_empty() {
            f.push((
                "info",
                "healthy",
                format!(
                    "No issues detected at iter {}: inf_pr={:.2e}, inf_du={:.2e}, μ={:.2e}.",
                    ctx.iter(),
                    inf_pr,
                    inf_du,
                    ctx.mu()
                ),
            ));
        }

        // Surface errors first, then warnings, then info.
        let rank = |s: &str| match s {
            "error" => 0,
            "warning" => 1,
            _ => 2,
        };
        f.sort_by_key(|(sev, _, _)| rank(sev));

        let lines: Vec<String> = f
            .iter()
            .map(|(sev, code, msg)| format!("[{sev:>7}] {code}: {msg}"))
            .collect();
        let data: Vec<_> = f
            .iter()
            .map(|(sev, code, msg)| serde_json::json!({"severity": sev, "code": code, "message": msg}))
            .collect();
        let n = data.len();
        CmdOut::ok(lines)
            .with_data(serde_json::json!({"iter": ctx.iter(), "findings": data, "n_findings": n}))
    }

    /// `print kkt` — inertia + regularization of the factored augmented
    /// system. Only meaningful at/after `after_search_dir`.
    fn cmd_print_kkt(&self, ctx: &dyn DebugState) -> CmdOut {
        let Some(k) = ctx.kkt() else {
            return CmdOut::err(
                "no KKT factorization yet — stop at `after_search_dir` (e.g. `stop-at kkt`)",
            );
        };
        let inertia = if k.provides_inertia {
            format!(
                "n+={} n-={} (expected n-={}) → {}",
                k.n_pos,
                k.n_neg,
                k.expected_neg,
                if k.inertia_correct {
                    "correct"
                } else {
                    "WRONG (step stabilized)"
                }
            )
        } else {
            "n/a (backend reports no inertia)".to_string()
        };
        let lines = vec![
            format!("dim       = {}", k.dim),
            format!("inertia   = {inertia}"),
            format!("delta_w   = {:.6e}   (primal regularization)", k.delta_w),
            format!("delta_c   = {:.6e}   (dual regularization)", k.delta_c),
            format!("status    = {}", k.status),
        ];
        CmdOut::ok(lines).with_data(serde_json::json!({
            "dim": k.dim,
            "n_pos": k.n_pos,
            "n_neg": k.n_neg,
            "expected_neg": k.expected_neg,
            "provides_inertia": k.provides_inertia,
            "inertia_correct": k.inertia_correct,
            "delta_w": k.delta_w,
            "delta_c": k.delta_c,
            "status": k.status,
        }))
    }

    /// `print rank` — numerical rank diagnosis of the equality-constraint
    /// Jacobian `J_c` at the current iterate. Runs a rank-revealing SVD,
    /// reports the numerical rank / condition number, and — when the block
    /// is rank-deficient — names the equations participating in the
    /// near-null space (the dependency the `δ_c` regularization is papering
    /// over). The numerical complement to the structural `diagnose` /
    /// Dulmage–Mendelsohn pass: it also catches *value* dependencies a
    /// full sparsity pattern hides.
    fn cmd_print_rank(&self, ctx: &DebugCtx) -> CmdOut {
        let Some(rep) = ctx.rank_report() else {
            return CmdOut::err(
                "no equality-constraint Jacobian to analyze (the problem has no equality \
                 constraints, or there is no iterate yet)",
            );
        };
        let names = ctx.split_names();
        let (lines, data) =
            render_rank_report(&rep, &names, self.equation_book.as_ref(), ctx.iter());
        CmdOut::ok(lines).with_data(data)
    }

    fn cmd_run(&mut self, rest: &[&str]) -> CmdOut {
        match rest.first().and_then(|s| s.parse::<i32>().ok()) {
            Some(n) => {
                self.run_to = Some(n);
                self.step = false;
                CmdOut::ok(vec![format!("running until iteration {n}")]).flow(Flow::Resume)
            }
            None => CmdOut::err("usage: run <iteration>"),
        }
    }

    fn cmd_break(&mut self, rest: &[&str]) -> CmdOut {
        // Conditional breakpoint: `break if <metric><op><value>`. Tokens
        // after `if` are concatenated so `inf_pr < 1e-6` and `inf_pr<1e-6`
        // parse the same.
        if rest.first().copied() == Some("if") {
            let expr: String = rest[1..].concat();
            if expr.is_empty() {
                return CmdOut::err(
                    "usage: break if <metric><op><value>  (e.g. break if inf_pr<1e-6)",
                );
            }
            return match Condition::parse(&expr) {
                Ok(c) => {
                    let raw = c.raw.clone();
                    if !self.conds.iter().any(|e| e.raw == raw) {
                        self.conds.push(c);
                    }
                    CmdOut::ok(vec![format!("conditional breakpoint: {raw}")])
                        .with_data(serde_json::json!({"condition": raw}))
                }
                Err(e) => CmdOut::err(e),
            };
        }
        // Event breakpoint: `break on <event>` (#72 §3).
        if rest.first().copied() == Some("on") {
            let Some(&name) = rest.get(1) else {
                return CmdOut::err(format!("usage: break on <event>  (one of {EVENTS:?})"));
            };
            let Some(&canon) = EVENTS.iter().find(|&&e| e == name) else {
                return CmdOut::err(format!("unknown event `{name}` (one of {EVENTS:?})"));
            };
            self.break_events.insert(canon);
            return CmdOut::ok(vec![format!("break on event `{canon}`")])
                .with_data(serde_json::json!({"event": canon}));
        }
        match rest {
            [] => {
                let mut bs = self.breaks.clone();
                bs.sort_unstable();
                let conds: Vec<String> = self.conds.iter().map(|c| c.raw.clone()).collect();
                let mut events: Vec<&str> = self.break_events.iter().copied().collect();
                events.sort_unstable();
                let mut lines = vec![format!("breakpoints: {bs:?}")];
                if !conds.is_empty() {
                    lines.push(format!("conditions: {}", conds.join(", ")));
                }
                if !events.is_empty() {
                    lines.push(format!("events: {}", events.join(", ")));
                }
                CmdOut::ok(lines).with_data(
                    serde_json::json!({"breakpoints": bs, "conditions": conds, "events": events}),
                )
            }
            ["clear", "cond"] | ["clear", "conditions"] => {
                self.conds.clear();
                CmdOut::ok(vec!["cleared conditional breakpoints".into()])
            }
            ["clear", "events"] => {
                self.break_events.clear();
                CmdOut::ok(vec!["cleared event breakpoints".into()])
            }
            ["clear"] => {
                self.breaks.clear();
                self.conds.clear();
                self.break_events.clear();
                CmdOut::ok(vec!["cleared all breakpoints".into()])
            }
            ["del", n] | ["delete", n] => match n.parse::<i32>() {
                Ok(n) => {
                    self.breaks.retain(|&b| b != n);
                    CmdOut::ok(vec![format!("removed breakpoint {n}")])
                }
                Err(_) => CmdOut::err("usage: break del <iteration>"),
            },
            [n] => match n.parse::<i32>() {
                Ok(n) => {
                    if !self.breaks.contains(&n) {
                        self.breaks.push(n);
                    }
                    CmdOut::ok(vec![format!("breakpoint at iteration {n}")])
                }
                Err(_) => CmdOut::err("usage: break <iteration>"),
            },
            _ => CmdOut::err("usage: break [N | if <m><op><v> | clear | clear cond | del N]"),
        }
    }

    /// `stop-at [name|clear]` — pause at a sub-iteration checkpoint every
    /// time it fires. Names: after_mu, after_search_dir, after_step
    /// (also iter_start / terminated). Aliases: mu, kkt/search_dir, step.
    fn cmd_stop_at(&mut self, rest: &[&str]) -> CmdOut {
        let canon = |s: &str| -> Option<&'static str> {
            match s {
                "mu" | "after_mu" => Some("after_mu"),
                "kkt" | "search_dir" | "after_search_dir" => Some("after_search_dir"),
                "step" | "after_step" => Some("after_step"),
                "rejected" | "ls_rejected" | "step_rejected" => Some("step_rejected"),
                "resto" | "restoration" | "pre_restoration_entry" => Some("pre_restoration_entry"),
                "resto_exit" | "post_restoration_exit" => Some("post_restoration_exit"),
                "iter" | "iter_start" => Some("iter_start"),
                "terminated" => Some("terminated"),
                _ => None,
            }
        };
        match rest {
            [] => {
                let mut v: Vec<&str> = self.stop_at.iter().copied().collect();
                v.sort_unstable();
                CmdOut::ok(vec![format!(
                    "stop-at: {v:?}  (available: {CHECKPOINTS:?})"
                )])
                .with_data(serde_json::json!({"stop_at": v, "available": CHECKPOINTS}))
            }
            ["clear"] => {
                self.stop_at.clear();
                CmdOut::ok(vec!["cleared stop-at checkpoints".into()])
            }
            [name] => match canon(name) {
                Some(c) => {
                    self.stop_at.insert(c);
                    CmdOut::ok(vec![format!("will stop at checkpoint `{c}`")])
                        .with_data(serde_json::json!({"stop_at_added": c}))
                }
                None => CmdOut::err(format!(
                    "unknown checkpoint `{name}` (one of {CHECKPOINTS:?})"
                )),
            },
            _ => CmdOut::err("usage: stop-at [<checkpoint> | clear]"),
        }
    }

    fn cmd_set(&mut self, rest: &[&str], ctx: &mut dyn DebugState) -> CmdOut {
        match rest {
            ["mu", v] => match v.parse::<f64>() {
                Ok(mu) => match ctx.set_mu(mu) {
                    Ok(()) => CmdOut::ok(vec![format!("mu := {mu:.6e}")]),
                    Err(e) => CmdOut::err(e),
                },
                Err(_) => CmdOut::err("usage: set mu <value>"),
            },
            ["opt", name, value] => match as_nlp_mut(ctx) {
                Some(c) => self.cmd_set_opt(name, value, c),
                None => nlp_only("set opt"),
            },
            [target, value] => self.cmd_set_block(target, value, ctx),
            _ => CmdOut::err(
                "usage: set mu <v> | set <blk>[<i>] <v> | set <blk> <v0,v1,..> | set opt <name> <v>",
            ),
        }
    }

    /// `set x[2] 1.5` (component) or `set x 1,2,3` (whole block).
    fn cmd_set_block(&mut self, target: &str, value: &str, ctx: &mut dyn DebugState) -> CmdOut {
        // Component form: name[idx]
        if let Some(open) = target.find('[') {
            if !target.ends_with(']') {
                return CmdOut::err("malformed component target (expected name[idx])");
            }
            let name = &target[..open];
            let idx_str = &target[open + 1..target.len() - 1];
            let Ok(idx) = idx_str.parse::<usize>() else {
                return CmdOut::err(format!("bad index `{idx_str}`"));
            };
            let Ok(val) = value.parse::<f64>() else {
                return CmdOut::err(format!("bad value `{value}`"));
            };
            return match ctx.set_component(name, idx, val) {
                Ok(()) => CmdOut::ok(vec![format!("{name}[{idx}] := {val:.6e}")]),
                Err(e) => CmdOut::err(e),
            };
        }
        // Whole-block form: comma-separated values.
        let parsed: Result<Vec<f64>, _> =
            value.split(',').map(|s| s.trim().parse::<f64>()).collect();
        match parsed {
            Ok(vals) => match ctx.set_block(target, &vals) {
                Ok(()) => CmdOut::ok(vec![format!("{target} := {} value(s)", vals.len())]),
                Err(e) => CmdOut::err(e),
            },
            Err(_) => CmdOut::err("could not parse comma-separated values"),
        }
    }

    fn cmd_set_opt(&mut self, name: &str, value: &str, ctx: &mut DebugCtx) -> CmdOut {
        let Some(reg) = self.reg.as_ref() else {
            return CmdOut::err("no options registry available");
        };
        let Some(opt) = reg.get_option(name) else {
            return CmdOut::err(format!("unknown option `{name}` (try `opt {name}`)"));
        };
        // Validate against the registered type/bounds.
        let valid = match opt.option_type {
            OptionType::OT_Number => value
                .parse::<f64>()
                .map(|v| opt.is_valid_number(v))
                .unwrap_or(false),
            OptionType::OT_Integer => value
                .parse::<i32>()
                .map(|v| opt.is_valid_integer(v))
                .unwrap_or(false),
            OptionType::OT_String => opt.is_valid_string(value),
            OptionType::OT_Unknown => true,
        };
        if !valid {
            return CmdOut::err(format!("`{value}` is not a valid value for `{name}`"));
        }
        // Record it on the staged list either way, so `get opt` reflects
        // it and a later `resolve` re-applies it from scratch.
        self.staged.retain(|(k, _)| k != name);
        self.staged.push((name.to_string(), value.to_string()));
        // Convergence tolerances are re-read by the conv-check policy each
        // iteration, so we can hot-swap them in place: hand the value to
        // the live `DebugCtx`, which the main loop drains after this hook
        // returns. The next `step` honors it — no `resolve` required.
        if is_live_tolerance(name) {
            if let Ok(v) = value.parse::<f64>() {
                ctx.set_live_tolerance(name, v);
                return CmdOut::ok(vec![format!(
                    "{name} = {value}  (applied live — the next `step` uses it)"
                )])
                .with_data(serde_json::json!({
                    "option": name, "value": value, "live": true
                }));
            }
        }
        CmdOut::ok(vec![format!(
            "staged {name} = {value}  (validated; takes effect on `resolve` — built strategies don't re-read mid-solve)"
        )])
        .with_data(serde_json::json!({"option": name, "value": value, "staged": true}))
    }

    /// `get opt <name>` (or the shorthand `get <name>`) — show the value
    /// an option would take on the next solve: the value you staged this
    /// session with `set opt`, if any, else the registered default. The
    /// debugger holds the staged overrides and the option registry, not
    /// the running solver's live `OptionsList`, so this is the *configured*
    /// value, not a mid-solve internal.
    fn cmd_get(&self, rest: &[&str]) -> CmdOut {
        // Accept both `get opt <name>` and the shorthand `get <name>`.
        let name = match rest {
            ["opt", n] => *n,
            [n] => *n,
            _ => return CmdOut::err("usage: get opt <name>"),
        };
        let Some(reg) = self.reg.as_ref() else {
            return CmdOut::err("no options registry available");
        };
        let Some(o) = reg.get_option(name) else {
            return CmdOut::err(format!("unknown option `{name}` (try `opt {name}`)"));
        };
        let def = default_str(&o.default);
        let staged = self
            .staged
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone());
        let (value, source) = match &staged {
            Some(v) => (v.clone(), "staged"),
            None => (def.clone(), "default"),
        };
        CmdOut::ok(vec![format!("{name} = {value}  ({source}; default={def})")]).with_data(
            serde_json::json!({
                "option": name, "value": value, "source": source,
                "default": def, "staged": staged,
            }),
        )
    }

    fn cmd_opt(&self, rest: &[&str]) -> CmdOut {
        let Some(reg) = self.reg.as_ref() else {
            return CmdOut::err("no options registry available");
        };
        let filter = rest.first().copied().unwrap_or("");
        let mut lines = Vec::new();
        let mut data = Vec::new();
        for o in reg.registered_options_in_order() {
            if !filter.is_empty()
                && !o.name.contains(filter)
                && !o
                    .category
                    .to_ascii_lowercase()
                    .contains(&filter.to_ascii_lowercase())
            {
                continue;
            }
            let ty = type_str(o.option_type);
            let def = default_str(&o.default);
            lines.push(format!(
                "  {:<28} {:<7} default={:<12} {}",
                o.name, ty, def, o.short_description
            ));
            data.push(serde_json::json!({
                "name": o.name,
                "type": ty,
                "default": def,
                "category": o.category,
                "short": o.short_description,
                "valid": o.valid_strings.iter().map(|e| e.value.clone()).collect::<Vec<_>>(),
            }));
        }
        if lines.is_empty() {
            return CmdOut::ok(vec![format!("no options match `{filter}`")]);
        }
        // For a single exact match, also show the long description.
        if data.len() == 1 {
            if let Some(o) = reg.get_option(filter) {
                if !o.long_description.is_empty() {
                    lines.push(String::new());
                    lines.push(o.long_description.clone());
                }
            }
        }
        CmdOut::ok(lines).with_data(serde_json::json!({"options": data}))
    }

    /// `complete <line…>` — context-sensitive completion candidates for
    /// the last token, using the same engine as TTY Tab. The preceding
    /// tokens form the context (so `complete set opt mu` completes option
    /// names, `complete set opt mu_strategy a` completes valid values).
    fn cmd_complete(&self, rest: &[&str]) -> CmdOut {
        let (before, word) = match rest.split_last() {
            Some((w, pre)) => (pre.join(" "), *w),
            None => (String::new(), ""),
        };
        let mut cands = completion_candidates(self.reg.as_deref(), &before, word);
        cands.sort();
        cands.dedup();
        CmdOut::ok(vec![cands.join(" ")]).with_data(serde_json::json!({"candidates": cands}))
    }

    /// `save [path]` — dump the full current iterate (all blocks +
    /// search-direction blocks) and residual scalars to a JSON file for
    /// external analysis. Defaults to a temp path keyed by iteration.
    fn cmd_save(&self, rest: &[&str], ctx: &dyn DebugState) -> CmdOut {
        let iter = ctx.iter();
        let path = rest
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join(format!("pounce-dbg-iter{iter}.json")));
        let collect = |delta: bool| -> serde_json::Map<String, serde_json::Value> {
            let mut m = serde_json::Map::new();
            for b in block_names(ctx) {
                let v = if delta {
                    ctx.delta_block(b)
                } else {
                    ctx.block(b)
                };
                if let Some(v) = v {
                    if !v.is_empty() {
                        let key = if delta {
                            format!("d{b}")
                        } else {
                            b.to_string()
                        };
                        m.insert(key, serde_json::json!(v));
                    }
                }
            }
            m
        };
        let payload = serde_json::json!({
            "iter": iter,
            "mu": ctx.mu(),
            "objective": ctx.objective(),
            "inf_pr": ctx.inf_pr(),
            "inf_du": ctx.inf_du(),
            "nlp_error": ctx.nlp_error(),
            "iterate": collect(false),
            "delta": collect(true),
        });
        match std::fs::write(&path, format!("{payload}\n")) {
            Ok(()) => {
                let p = path.to_string_lossy().to_string();
                CmdOut::ok(vec![format!("saved iterate to {p}")])
                    .with_data(serde_json::json!({"path": p}))
            }
            Err(e) => CmdOut::err(format!("save failed: {e}")),
        }
    }

    /// `load <file> [block]` — the inverse of `save`. Read a block (by
    /// default `x`) into the live iterate from either a `save` artifact
    /// (JSON: top-level or under `iterate`, every block found is loaded) or
    /// a plain numeric file (comma/whitespace/newline-separated values →
    /// the named block, default `x`). The point that a many-variable start
    /// is awkward to type by hand — generate it once, `load` it here.
    fn cmd_load(&mut self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
        let Some(&path) = rest.first() else {
            return CmdOut::err("usage: load <file> [block]   (inverse of `save`)");
        };
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return CmdOut::err(format!("cannot read `{path}`: {e}")),
        };
        // JSON path: a `save` artifact (blocks at top level or under
        // `iterate`). Load every block present; report dims and any
        // dimension mismatches per block.
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(content.trim()) {
            let obj = v
                .get("iterate")
                .and_then(|o| o.as_object())
                .or_else(|| v.as_object());
            if let Some(obj) = obj {
                let mut loaded: Vec<(String, usize)> = Vec::new();
                let mut errs: Vec<String> = Vec::new();
                for &b in BLOCK_NAMES.iter() {
                    let Some(arr) = obj.get(b).and_then(|a| a.as_array()) else {
                        continue;
                    };
                    let vals: Option<Vec<f64>> = arr.iter().map(|x| x.as_f64()).collect();
                    let Some(vals) = vals else {
                        errs.push(format!("{b}: non-numeric entries"));
                        continue;
                    };
                    match ctx.set_block(b, &vals) {
                        Ok(()) => loaded.push((b.to_string(), vals.len())),
                        Err(e) => errs.push(format!("{b}: {e}")),
                    }
                }
                if loaded.is_empty() && errs.is_empty() {
                    return CmdOut::err(
                        "no recognizable blocks in JSON (expected `x`, `s`, … at top level or under `iterate`)",
                    );
                }
                let mut lines: Vec<String> = loaded
                    .iter()
                    .map(|(b, n)| format!("loaded {b} ({n} values)"))
                    .collect();
                lines.extend(errs.iter().map(|e| format!("skipped {e}")));
                return CmdOut::ok(lines).with_data(serde_json::json!({
                    "loaded": loaded.iter().map(|(b, n)| serde_json::json!({"block": b, "n": n})).collect::<Vec<_>>(),
                    "skipped": errs,
                }));
            }
        }
        // Raw numeric path: parse floats and set the named block (default x).
        let block = rest.get(1).copied().unwrap_or("x");
        let vals = match parse_floats(&content) {
            Ok(v) if !v.is_empty() => v,
            Ok(_) => return CmdOut::err("file held no numbers"),
            Err(e) => return CmdOut::err(e),
        };
        match ctx.set_block(block, &vals) {
            Ok(()) => CmdOut::ok(vec![format!("loaded {block} ({} values)", vals.len())])
                .with_data(serde_json::json!({"block": block, "n": vals.len()})),
            Err(e) => CmdOut::err(e),
        }
    }

    /// `sweep <file>` — run one full solve per start point in `file` (one
    /// start per line, comma/whitespace-separated; `#` comments skipped),
    /// then tabulate the terminal status / objective of each. An
    /// initialization-sensitivity probe: which starts converge, and to
    /// which minima. Needs the re-solve machinery (a restart cell).
    fn cmd_sweep(&mut self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
        if self.restart.is_none() {
            return CmdOut::err("sweep needs re-solve, which is not available in this context");
        }
        let Some(&path) = rest.first() else {
            return CmdOut::err("usage: sweep <file>   (one start per line, comma-separated)");
        };
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return CmdOut::err(format!("cannot read `{path}`: {e}")),
        };
        let dim = ctx.block("x").map(|x| x.len()).unwrap_or(0);
        let mut seeds: Vec<Vec<f64>> = Vec::new();
        for (lineno, raw) in content.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            match parse_floats(line) {
                Ok(v) if v.len() == dim => seeds.push(v),
                Ok(v) => {
                    return CmdOut::err(format!(
                        "line {}: got {} values, expected {dim} (= dim x)",
                        lineno + 1,
                        v.len()
                    ));
                }
                Err(e) => return CmdOut::err(format!("line {}: {e}", lineno + 1)),
            }
        }
        self.start_sweep(seeds, &format!("sweep `{path}`"))
    }

    /// `multistart <N> [rel]` — run `N` full solves from sampled starts,
    /// then tabulate the outcomes. Each variable with a finite box
    /// `[x_Lᵢ, x_Uᵢ]` is sampled **uniformly in that box**; variables that
    /// are unbounded on either side fall back to a relative jitter
    /// `±rel·(|xᵢ|+1)` around the current point (`rel` default 0.1). Start 0
    /// is always the current `x`. Deterministic (a fixed-seed PRNG), so runs
    /// reproduce.
    fn cmd_multistart(&mut self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
        if self.restart.is_none() {
            return CmdOut::err(
                "multistart needs re-solve, which is not available in this context",
            );
        }
        let Some(n) = rest.first().and_then(|s| s.parse::<usize>().ok()) else {
            return CmdOut::err("usage: multistart <N> [rel]   (N sampled restarts)");
        };
        if n == 0 {
            return CmdOut::err("N must be ≥ 1");
        }
        let rel = rest
            .get(1)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.1);
        let Some(base) = ctx.block("x") else {
            return CmdOut::err("no current iterate to sample from");
        };
        // Full-length algorithm-space bounds, if available and aligned.
        let bounds = ctx
            .var_bounds()
            .filter(|(lo, hi)| lo.len() == base.len() && hi.len() == base.len());
        let n_box = bounds
            .as_ref()
            .map(|(lo, hi)| {
                lo.iter()
                    .zip(hi)
                    .filter(|(l, u)| l.is_finite() && u.is_finite() && u > l)
                    .count()
            })
            .unwrap_or(0);
        let seeds: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let b = bounds
                    .as_ref()
                    .map(|(lo, hi)| (lo.as_slice(), hi.as_slice()));
                sample_start(&base, b, rel, k)
            })
            .collect();
        let n_var = base.len();
        let label = if n_box == n_var {
            format!("multistart {n} (box-sampled, {n_box}/{n_var} vars bounded)")
        } else if n_box > 0 {
            format!(
                "multistart {n} (box {n_box}/{n_var} vars; {} unbounded → jitter rel={rel})",
                n_var - n_box
            )
        } else {
            format!("multistart {n} (no finite boxes → jitter rel={rel})")
        };
        self.start_sweep(seeds, &label)
    }

    /// Launch a sweep: stop the current solve and re-solve from the first
    /// seed; the rest are driven from the terminal checkpoint
    /// ([`Self::drive_sweep`]). Each solve runs free (`pause_iters` off,
    /// restored when the sweep ends).
    fn start_sweep(&mut self, seeds: Vec<Vec<f64>>, label: &str) -> CmdOut {
        if seeds.is_empty() {
            return CmdOut::err("no start points");
        }
        let Some(cell) = self.restart.as_ref() else {
            return CmdOut::err("sweep needs re-solve, which is not available in this context");
        };
        let total = seeds.len();
        let mut queue: VecDeque<Vec<f64>> = seeds.into();
        let first = queue.pop_front().expect("non-empty");
        *cell.borrow_mut() = Some(RestartRequest {
            seed_x: first.clone(),
            options: self.staged.clone(),
            warm: None,
        });
        // Run each sweep solve free; we intercept only at the terminal
        // checkpoint. Clear any one-shot arming so the re-solve doesn't pause.
        let saved_pause_iters = self.pause_iters;
        self.pause_iters = false;
        self.step = false;
        self.sub_step = false;
        self.run_to = None;
        self.sweep = Some(SweepState {
            queue,
            current: Some(first),
            records: Vec::new(),
            total,
            saved_pause_iters,
        });
        CmdOut::ok(vec![format!("{label}: running {total} start(s)…")])
            .with_data(serde_json::json!({"sweep": label, "starts": total}))
            .flow(Flow::Stop)
    }

    /// Drive an in-flight sweep at the terminal checkpoint: record the
    /// solve that just finished, then either launch the next seed (returns
    /// `Some(Resume)` — the CLI re-solve loop picks up the queued
    /// [`RestartRequest`]) or, when the queue drains, print the summary,
    /// restore state, and return `None` so the caller falls through to the
    /// normal terminal handling.
    fn drive_sweep(&mut self, ctx: &DebugCtx) -> Option<DebugAction> {
        let mut sweep = self.sweep.take()?;
        let rec = SweepRecord {
            idx: sweep.records.len(),
            seed: sweep.current.clone().unwrap_or_default(),
            status: ctx.status().unwrap_or("?").to_string(),
            objective: ctx.objective(),
            inf_pr: ctx.inf_pr(),
            iters: ctx.iter(),
        };
        self.emit_sweep_progress(&rec, sweep.total);
        sweep.records.push(rec);
        if let Some(next) = sweep.queue.pop_front() {
            sweep.current = Some(next.clone());
            if let Some(cell) = self.restart.as_ref() {
                *cell.borrow_mut() = Some(RestartRequest {
                    seed_x: next,
                    options: self.staged.clone(),
                    warm: None,
                });
            }
            self.sweep = Some(sweep);
            return Some(DebugAction::Resume);
        }
        // Sweep complete: restore per-iteration pausing and report.
        self.pause_iters = sweep.saved_pause_iters;
        self.emit_sweep_summary(&sweep);
        None
    }

    /// One-line-per-solve progress as a sweep runs (REPL → stderr; JSON →
    /// a `sweep_result` event).
    fn emit_sweep_progress(&self, rec: &SweepRecord, total: usize) {
        match self.mode {
            DebugMode::Repl => eprintln!(
                "   sweep {}/{}: {:<22} iters={:<4} obj={:.6e} inf_pr={:.2e}",
                rec.idx + 1,
                total,
                rec.status,
                rec.iters,
                rec.objective,
                rec.inf_pr,
            ),
            DebugMode::Json => emit_json(&serde_json::json!({
                "event": "sweep_result",
                "index": rec.idx,
                "total": total,
                "status": rec.status,
                "iters": rec.iters,
                "objective": rec.objective,
                "inf_pr": rec.inf_pr,
                "seed": rec.seed,
            })),
        }
    }

    /// Final sweep summary: a table of every solve plus a distinct-minima
    /// count and the best (lowest-objective) successful solve.
    fn emit_sweep_summary(&self, sweep: &SweepState) {
        let succeeded: Vec<&SweepRecord> = sweep
            .records
            .iter()
            .filter(|r| is_success_status(&r.status))
            .collect();
        // Distinct minima: successful objectives clustered to a relative 1e-6.
        let mut distinct: Vec<f64> = Vec::new();
        for r in &succeeded {
            if !distinct
                .iter()
                .any(|&o| (o - r.objective).abs() <= 1e-6 * o.abs().max(1.0))
            {
                distinct.push(r.objective);
            }
        }
        let best = succeeded.iter().min_by(|a, b| {
            a.objective
                .partial_cmp(&b.objective)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        match self.mode {
            DebugMode::Repl => {
                eprintln!(
                    "\n── sweep complete ── {} solves, {} succeeded, {} distinct minima",
                    sweep.records.len(),
                    succeeded.len(),
                    distinct.len()
                );
                eprintln!(
                    "   {:>3}  {:<22} {:>5}  {:>14}  {:>9}",
                    "#", "status", "iters", "objective", "inf_pr"
                );
                for r in &sweep.records {
                    eprintln!(
                        "   {:>3}  {:<22} {:>5}  {:>14.6e}  {:>9.2e}",
                        r.idx, r.status, r.iters, r.objective, r.inf_pr
                    );
                }
                if let Some(b) = best {
                    eprintln!("   best: solve #{}  obj={:.8e}", b.idx, b.objective);
                }
            }
            DebugMode::Json => emit_json(&serde_json::json!({
                "event": "sweep_summary",
                "solves": sweep.records.len(),
                "succeeded": succeeded.len(),
                "distinct_minima": distinct.len(),
                "best_index": best.map(|b| b.idx),
                "best_objective": best.map(|b| b.objective),
                "records": sweep.records.iter().map(|r| serde_json::json!({
                    "index": r.idx, "status": r.status, "iters": r.iters,
                    "objective": r.objective, "inf_pr": r.inf_pr,
                })).collect::<Vec<_>>(),
            })),
        }
    }

    /// `goto <k>` — rewind to a captured iteration.
    fn cmd_goto(&mut self, rest: &[&str], ctx: &mut dyn DebugState) -> CmdOut {
        match rest.first().and_then(|s| s.parse::<i32>().ok()) {
            Some(k) => self.restore_to(k, ctx),
            None => CmdOut::err("usage: goto <iteration>"),
        }
    }

    /// Restore the snapshot for iteration `k` (primal-dual state only;
    /// strategy history is not rewound). Stays paused so the user can
    /// inspect / re-tune before `continue`/`step`.
    fn restore_to(&mut self, k: i32, ctx: &mut dyn DebugState) -> CmdOut {
        match self.snapshots.get(&k) {
            Some(snap) => {
                if !ctx.restore(snap.as_ref()) {
                    return CmdOut::err(format!(
                        "this solver does not support rewinding to iter {k}"
                    ));
                }
                CmdOut::ok(vec![format!(
                    "rewound to iter {k} (primal-dual only; strategy history not restored). \
                     `continue`/`step` to resume."
                )])
                .with_data(serde_json::json!({"restored_iter": k}))
            }
            None => {
                let have: Vec<i32> = self.snapshots.keys().copied().collect();
                CmdOut::err(format!("no snapshot for iter {k} (captured: {have:?})"))
            }
        }
    }

    /// `resolve` — capture the full primal-dual iterate (all 8 blocks +
    /// μ) and the staged option edits, then stop this solve so the CLI
    /// re-runs continuing from that interior point with the new options
    /// applied (a true warm start: duals carry over, the barrier resumes
    /// at the current μ rather than restarting at `mu_init`). Falls back
    /// to a primal-only seed if the iterate can't be snapshotted. Needs a
    /// restart cell (wired by the CLI); a no-op error otherwise.
    fn cmd_resolve(&mut self, ctx: &DebugCtx) -> CmdOut {
        let Some(cell) = self.restart.as_ref() else {
            return CmdOut::err("re-solve is not available in this context");
        };
        let Some(seed_x) = ctx.block("x") else {
            return CmdOut::err("no current iterate to seed from");
        };
        let warm = ctx.snapshot();
        let mu = warm.as_ref().map(|s| s.mu());
        let options = self.staged.clone();
        let n_opt = options.len();
        let warm_msg = match mu {
            Some(mu) => format!(
                "re-solving warm from the current primal-dual iterate (μ={mu:.3e}) \
                 with {n_opt} staged option override(s)…"
            ),
            None => format!(
                "re-solving from current x (primal-only) with {n_opt} staged option override(s)…"
            ),
        };
        *cell.borrow_mut() = Some(RestartRequest {
            seed_x,
            options,
            warm,
        });
        CmdOut::ok(vec![warm_msg])
            .with_data(serde_json::json!({
                "resolve": true,
                "options": n_opt,
                "warm": mu.is_some(),
                "mu": mu,
            }))
            .flow(Flow::Stop)
    }

    /// `ask [question]` — hand the current solver state to Claude Code
    /// (headless `claude -p`, or `$POUNCE_DBG_LLM`) and print its reply.
    /// "Ask why this step looks wrong without leaving the debugger."
    fn cmd_ask(&self, rest: &[&str], ctx: &dyn DebugState) -> CmdOut {
        let question = if rest.is_empty() {
            "Explain the current state of this interior-point solve and suggest what to try next."
                .to_string()
        } else {
            rest.join(" ")
        };
        let prompt = build_ask_prompt(ctx, &question);
        match run_llm(&prompt) {
            Ok(reply) => {
                let lines: Vec<String> = reply.lines().map(|l| l.to_string()).collect();
                CmdOut::ok(lines).with_data(serde_json::json!({
                    "question": question,
                    "reply": reply,
                }))
            }
            Err(e) => CmdOut::err(e),
        }
    }

    /// `watch [target|clear|del <target>]` — auto-print a `print` target
    /// (block, `dx`, scalar, `kkt`) at every pause.
    fn cmd_watch(&mut self, rest: &[&str]) -> CmdOut {
        match rest {
            [] => CmdOut::ok(vec![format!("watches: {:?}", self.watches)])
                .with_data(serde_json::json!({"watches": self.watches})),
            ["clear"] => {
                self.watches.clear();
                CmdOut::ok(vec!["cleared watches".into()])
            }
            ["del", w] | ["delete", w] => {
                self.watches.retain(|x| x != w);
                CmdOut::ok(vec![format!("unwatched {w}")])
            }
            [w] => {
                let w = w.to_string();
                if !self.watches.contains(&w) {
                    self.watches.push(w.clone());
                }
                CmdOut::ok(vec![format!("watching {w}")])
            }
            _ => CmdOut::err("usage: watch [<target> | clear | del <target>]"),
        }
    }

    /// `watchpoint <blk>[<i>] [threshold] | clear | del <spec>` — pause
    /// when a watched value changes by more than `threshold` (default 0,
    /// any change) between iterations.
    fn cmd_watchpoint(&mut self, rest: &[&str], ctx: &dyn DebugState) -> CmdOut {
        match rest {
            [] => {
                let v: Vec<&str> = self.watchpoints.iter().map(|w| w.raw.as_str()).collect();
                CmdOut::ok(vec![format!("watchpoints: {v:?}")])
                    .with_data(serde_json::json!({"watchpoints": v}))
            }
            ["clear"] => {
                self.watchpoints.clear();
                CmdOut::ok(vec!["cleared watchpoints".into()])
            }
            ["del", spec] | ["delete", spec] => {
                self.watchpoints.retain(|w| w.raw != *spec);
                CmdOut::ok(vec![format!("removed watchpoint {spec}")])
            }
            [spec, rest @ ..] => {
                let threshold = rest
                    .first()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                // Parse `block` or `block[idx]`.
                let (block, idx) = match spec.find('[') {
                    Some(open) if spec.ends_with(']') => {
                        let b = &spec[..open];
                        match spec[open + 1..spec.len() - 1].parse::<usize>() {
                            Ok(i) => (b.to_string(), Some(i)),
                            Err(_) => return CmdOut::err(format!("bad index in `{spec}`")),
                        }
                    }
                    _ => (spec.to_string(), None),
                };
                if !is_block(ctx, block.as_str()) {
                    return CmdOut::err(format!("unknown block `{block}`"));
                }
                let raw = spec.to_string();
                if !self.watchpoints.iter().any(|w| w.raw == raw) {
                    self.watchpoints.push(WatchPoint {
                        raw: raw.clone(),
                        block,
                        idx,
                        threshold,
                        last: None,
                    });
                }
                CmdOut::ok(vec![format!("watchpoint on {raw} (Δ>{threshold:.3e})")])
            }
        }
    }

    /// `commands <iter> <cmd> ; <cmd> …` — attach an auto-run command
    /// list to the breakpoint at iteration `iter` (e.g.
    /// `commands 5 set mu 0.1 ; continue`). `commands <iter> clear`
    /// removes it; `commands` lists all.
    fn cmd_commands(&mut self, rest: &[&str]) -> CmdOut {
        let Some(iter) = rest.first().and_then(|s| s.parse::<i32>().ok()) else {
            if rest.is_empty() {
                let mut items: Vec<(i32, Vec<String>)> = self
                    .bp_commands
                    .iter()
                    .map(|(k, v)| (*k, v.clone()))
                    .collect();
                items.sort_by_key(|(k, _)| *k);
                let lines = if items.is_empty() {
                    vec!["no breakpoint command lists".into()]
                } else {
                    items
                        .iter()
                        .map(|(k, v)| format!("iter {k}: {}", v.join(" ; ")))
                        .collect()
                };
                return CmdOut::ok(lines);
            }
            return CmdOut::err(
                "usage: commands <iter> <cmd> ; <cmd> …  (or: commands <iter> clear)",
            );
        };
        let tail = rest[1..].join(" ");
        let tail = tail.trim();
        if tail.is_empty() || tail == "clear" {
            self.bp_commands.remove(&iter);
            return CmdOut::ok(vec![format!("cleared commands for iteration {iter}")]);
        }
        let cmds: Vec<String> = tail
            .split(';')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        self.bp_commands.insert(iter, cmds.clone());
        CmdOut::ok(vec![format!(
            "commands for iter {iter}: {}",
            cmds.join(" ; ")
        )])
        .with_data(serde_json::json!({"iter": iter, "commands": cmds}))
    }

    /// `diff` — what changed in the iterate since the previous captured
    /// iteration: per-block max |Δ| (and where), plus Δμ.
    fn cmd_diff(&self, ctx: &dyn DebugState) -> CmdOut {
        let iter = ctx.iter();
        let Some((&piter, prev)) = self.snapshots.range(..iter).next_back() else {
            return CmdOut::err("no previous iterate to diff against");
        };
        let mut lines = vec![format!("Δ since iter {piter}:")];
        let dmu = ctx.mu() - prev.mu();
        lines.push(format!("  mu  = {:.6e}  (Δ {:+.3e})", ctx.mu(), dmu));
        let mut blocks = serde_json::Map::new();
        for b in block_names(ctx) {
            let (Some(cur), Some(old)) = (ctx.block(b), prev.block(b)) else {
                continue;
            };
            if cur.is_empty() || cur.len() != old.len() {
                continue;
            }
            let mut amax = 0.0_f64;
            let mut imax = 0usize;
            for (i, (c, o)) in cur.iter().zip(&old).enumerate() {
                let d = (c - o).abs();
                if d > amax {
                    amax = d;
                    imax = i;
                }
            }
            if amax > 0.0 {
                lines.push(format!(
                    "  {b}: max|Δ|={amax:.3e} at [{imax}]  ({:.4e} → {:.4e})",
                    old[imax], cur[imax]
                ));
                blocks.insert(
                    b.to_string(),
                    serde_json::json!({"max_abs_change": amax, "argmax": imax}),
                );
            }
        }
        if lines.len() == 2 {
            lines.push("  (no change)".into());
        }
        CmdOut::ok(lines).with_data(
            serde_json::json!({"from_iter": piter, "to_iter": iter, "dmu": dmu, "blocks": blocks}),
        )
    }

    /// `source <file>` — run debugger commands from a file (one per line;
    /// `#` comments and blank lines skipped). Stops early if a command
    /// resumes or stops the solve, propagating that control flow.
    fn cmd_source(&mut self, rest: &[&str], ctx: &mut dyn DebugState) -> CmdOut {
        let Some(&path) = rest.first() else {
            return CmdOut::err("usage: source <file>");
        };
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return CmdOut::err(format!("cannot read `{path}`: {e}")),
        };
        let mut lines = Vec::new();
        let mut flow = Flow::Stay;
        for raw in content.lines() {
            let cmd = raw.trim();
            if cmd.is_empty() || cmd.starts_with('#') || cmd.starts_with("//") {
                continue;
            }
            lines.push(format!("[source] {cmd}"));
            let out = self.dispatch(cmd, ctx);
            lines.extend(out.lines);
            if !matches!(out.flow, Flow::Stay) {
                flow = out.flow;
                break;
            }
        }
        CmdOut {
            ok: true,
            lines,
            data: None,
            flow,
        }
    }

    fn cmd_viz(&self, rest: &[&str], ctx: &mut dyn DebugState) -> CmdOut {
        let Some(&target) = rest.first() else {
            return CmdOut::err("usage: viz <x|s|y_c|...|dx|kkt|L>");
        };
        // `viz kkt` writes the assembled augmented-system matrix (triplets
        // → heatmap) plus the inertia/regularization summary.
        if target == "kkt" {
            let Some(k) = ctx.kkt() else {
                return CmdOut::err(
                    "no KKT factorization captured yet — nothing has been factored (iter 0), \
                     or the debugger is detached. `step` once to capture.",
                );
            };
            // The matrix triplets are captured into `kkt_debug` whenever the
            // debugger is stepping, so once anything has been factored they're
            // here — this is the previous iteration's system at `iter_start`,
            // the current one at `after_search_dir`.
            let Some((dim, irn, jcn, vals)) = ctx.kkt_matrix() else {
                return CmdOut::err(
                    "KKT matrix not captured here — the debugger is detached \
                     (running free). `step` once to capture and re-run `viz kkt`.",
                );
            };
            // Label with the iteration the factorization came from — at an
            // `iter_start` pause that's the previous iteration, not `ctx.iter()`.
            let kiter = k.iter;
            let matrix = serde_json::json!({"dim": dim, "irn": irn, "jcn": jcn, "vals": vals,
                                            "format": "triplet_1based_lower"});
            let payload = serde_json::json!({
                "label": "kkt", "iter": kiter,
                "dim": k.dim, "n_pos": k.n_pos, "n_neg": k.n_neg,
                "expected_neg": k.expected_neg, "inertia_correct": k.inertia_correct,
                "delta_w": k.delta_w, "delta_c": k.delta_c, "status": k.status,
                "matrix": matrix,
            });
            return match write_json_and_open("kkt", kiter, &payload) {
                Ok((path, viewer)) => CmdOut::ok(vec![format!(
                    "wrote {path} (KKT system, iter {kiter}); opened with `{viewer}`"
                )])
                .with_data(serde_json::json!({"path": path, "viewer": viewer})),
                Err(e) => CmdOut::err(e),
            };
        }
        // `viz L` writes the LDLᵀ factor triplets, read out of the factor
        // the solver actually computed. Captured into `kkt_debug` whenever
        // the debugger is stepping (same as the matrix), so it shows the
        // previous iteration's factorization at `iter_start`.
        if target == "L" {
            match ctx.kkt_l_factor() {
                Some((n, perm, l_irn, l_jcn, l_vals)) => {
                    // Iteration the factor came from (previous iter at `iter_start`).
                    let kiter = ctx.kkt_captured_iter().unwrap_or_else(|| ctx.iter());
                    let payload = serde_json::json!({
                        "label": "L", "iter": kiter, "n": n, "perm": perm,
                        "l_irn": l_irn, "l_jcn": l_jcn, "l_vals": l_vals,
                        "format": "strict_lower_1based_permuted",
                    });
                    return match write_json_and_open("L", kiter, &payload) {
                        Ok((path, viewer)) => CmdOut::ok(vec![format!(
                            "wrote {path} (L factor, iter {kiter}); opened with `{viewer}`"
                        )])
                        .with_data(serde_json::json!({"path": path, "viewer": viewer})),
                        Err(e) => CmdOut::err(e),
                    };
                }
                None => {
                    return CmdOut::err(
                        "L factor not captured here — nothing factored yet (iter 0), \
                         or the debugger is detached. `step` once to capture.",
                    );
                }
            }
        }
        // Resolve the vector to visualize.
        let (label, vals) = if is_block(ctx, target) {
            match ctx.block(target) {
                Some(v) => (target.to_string(), v),
                None => return CmdOut::err(format!("no data for block `{target}`")),
            }
        } else if let Some(blk) = target.strip_prefix("d").filter(|b| is_block(ctx, b)) {
            match ctx.delta_block(blk) {
                Some(v) => (format!("d{blk}"), v),
                None => return CmdOut::err(format!("no search direction for `d{blk}`")),
            }
        } else {
            return CmdOut::err(format!("don't know how to visualize `{target}`"));
        };
        match write_and_open(&label, ctx.iter(), &vals) {
            Ok((path, viewer)) => CmdOut::ok(vec![format!(
                "wrote {} ({} values); opened with `{}`",
                path,
                vals.len(),
                viewer
            )])
            .with_data(serde_json::json!({"path": path, "viewer": viewer, "n": vals.len()})),
            Err(e) => CmdOut::err(e),
        }
    }

    // ---- front ends ----------------------------------------------------

    /// Emit the pause banner / state for the current front end.
    fn emit_pause(&self, ctx: &dyn DebugState, reason: Option<&str>) {
        let terminal = matches!(ctx.checkpoint(), Checkpoint::Terminated);
        match self.mode {
            DebugMode::Repl => {
                if terminal {
                    eprintln!(
                        "\n── pounce-dbg ── TERMINATED ({})  iter {}  obj={:.6e}  inf_pr={:.2e}  inf_du={:.2e}",
                        ctx.status().unwrap_or("?"),
                        ctx.iter(),
                        ctx.objective(),
                        ctx.inf_pr(),
                        ctx.inf_du(),
                    );
                } else {
                    let resto = if self.in_restoration {
                        " [restoration]"
                    } else {
                        ""
                    };
                    eprintln!(
                        "\n── pounce-dbg ── iter {} @{}{}  mu={:.3e}  obj={:.6e}  inf_pr={:.2e}  inf_du={:.2e}",
                        ctx.iter(),
                        ctx.checkpoint().as_str(),
                        resto,
                        ctx.mu(),
                        ctx.objective(),
                        ctx.inf_pr(),
                        ctx.inf_du(),
                    );
                }
                if let Some(r) = reason {
                    eprintln!("   ↳ {r}");
                }
                for w in &self.watches {
                    let out = self.cmd_print(&[w.as_str()], ctx);
                    if out.ok {
                        for l in &out.lines {
                            eprintln!("   watch {l}");
                        }
                    } else {
                        // Don't spam the full error every pause for a target
                        // that isn't available yet (e.g. `kkt` before a
                        // factorization) — a compact note instead.
                        eprintln!("   watch {w}: (n/a)");
                    }
                }
            }
            DebugMode::Json => {
                let watches: Vec<serde_json::Value> = self
                    .watches
                    .iter()
                    .map(|w| {
                        let out = self.cmd_print(&[w.as_str()], ctx);
                        serde_json::json!({"expr": w, "ok": out.ok, "output": out.lines, "data": out.data})
                    })
                    .collect();
                let dims: serde_json::Map<String, serde_json::Value> = ctx
                    .block_dims()
                    .into_iter()
                    .map(|(n, d)| (n.to_string(), serde_json::json!(d)))
                    .collect();
                let conds: Vec<String> = self.conds.iter().map(|c| c.raw.clone()).collect();
                let ev = serde_json::json!({
                    "event": "pause",
                    "checkpoint": ctx.checkpoint().as_str(),
                    "status": ctx.status(),
                    "in_restoration": self.in_restoration,
                    "iter": ctx.iter(),
                    "mu": ctx.mu(),
                    "objective": ctx.objective(),
                    "inf_pr": ctx.inf_pr(),
                    "inf_du": ctx.inf_du(),
                    "nlp_error": ctx.nlp_error(),
                    "dims": dims,
                    "breakpoints": self.breaks,
                    "conditions": conds,
                    "reason": reason,
                    "watches": watches,
                });
                emit_json(&ev);
            }
        }
    }

    /// Emit a per-iteration `progress` event (JSON mode only). Same
    /// scalars as `pause`; fired while running between pauses.
    fn emit_progress_event(&self, ctx: &dyn DebugState) {
        let ev = serde_json::json!({
            "event": "progress",
            "iter": ctx.iter(),
            "mu": ctx.mu(),
            "inf_pr": ctx.inf_pr(),
            "inf_du": ctx.inf_du(),
            "obj": ctx.objective(),
        });
        emit_json(&ev);
    }

    /// Emit a command result for the current front end. `req_id` is the
    /// client's request id (JSON mode), echoed for response correlation.
    fn emit_result(&self, command: &str, out: &CmdOut, req_id: Option<&serde_json::Value>) {
        match self.mode {
            DebugMode::Repl => {
                let stderr = std::io::stderr();
                let mut h = stderr.lock();
                for l in &out.lines {
                    let _ = writeln!(h, "{l}");
                }
                if !out.ok {
                    let _ = writeln!(h, "(error)");
                }
            }
            DebugMode::Json => {
                let ev = serde_json::json!({
                    "event": "result",
                    "request_id": req_id,
                    "command": command,
                    "ok": out.ok,
                    "output": out.lines,
                    "data": out.data,
                });
                emit_json(&ev);
            }
        }
    }

    /// Emit the one-time JSON handshake: protocol version, the solver
    /// version, advertised capabilities, and the command / metric
    /// vocabulary — everything a visual debugger needs to configure its
    /// UI before the first `pause`.
    fn emit_hello(&self) {
        let ev = serde_json::json!({
            "event": "hello",
            "protocol": "pounce-dbg/1",
            "pounce_version": env!("CARGO_PKG_VERSION"),
            "capabilities": {
                "inspect": true,
                "mutate_iterate": true,
                "mutate_mu": true,
                "conditional_breakpoints": "compound",
                "request_ids": true,
                "viz": ["block", "delta"],
                "save": true,
                "load": true,
                "sweep": self.restart.is_some(),
                "kkt_inspect": true,
                // `print equation <name|row>` is available when a source
                // model (`.nl`) supplied constraint algebra to render.
                "equations": self.equation_book.is_some(),
                // Live `diagnose` — point-in-time named health findings.
                "diagnose": true,
                // `diagnose`'s structural rank pass (Dulmage–Mendelsohn)
                // names dependent equations; available with a `.nl` model.
                "structural_diagnose": self.structure_book.is_some(),
                "llm_assist": true,
                "rewind": "primal_dual",
                "resolve": self.restart.is_some(),
                "terminal_checkpoint": true,
                "interruptible": self.interruptible,
                // #72 §1 / §5.
                "progress_events": self.emit_progress,
                "async_pause": "checkpoint",
                // Both transports for async pause: SIGINT and the in-band
                // `{"cmd":"pause"}` (JSON mode).
                "pause_command": true,
            },
            "checkpoints": CHECKPOINTS,
            "events": EVENTS,
            "commands": COMMANDS,
            "blocks": BLOCK_NAMES,
            "metrics": METRICS,
        });
        emit_json(&ev);
    }

    /// Lazily build the rustyline editor for an interactive REPL on a
    /// TTY. No-op for JSON mode, non-terminal stdin, or if construction
    /// fails — those paths fall back to a plain line reader.
    fn ensure_editor(&mut self) {
        if !matches!(self.mode, DebugMode::Repl)
            || self.editor.is_some()
            || !std::io::stdin().is_terminal()
        {
            return;
        }
        let mut ed: Editor<DbgHelper, FileHistory> = match Editor::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        ed.set_helper(Some(DbgHelper {
            reg: self.reg.clone(),
        }));
        let path = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|h| PathBuf::from(h).join(".pounce_dbg_history"));
        if let Some(p) = &path {
            let _ = ed.load_history(p);
        }
        self.hist_path = path;
        self.editor = Some(ed);
    }

    /// Handle a Ctrl-C received at the prompt. Returns the command line to
    /// feed the loop: the first interrupt in a row cancels the line (empty
    /// string → reprompt) with a hint; a second quits the solve. The
    /// counter resets when any real line is entered (see `next_command_line`).
    fn on_prompt_interrupt(&mut self) -> String {
        self.prompt_interrupts += 1;
        if self.prompt_interrupts >= 2 {
            self.prompt_interrupts = 0;
            eprintln!("(quitting — Ctrl-C)");
            "quit".to_string()
        } else {
            eprintln!("(Ctrl-C — press again, or `quit`/Ctrl-D, to stop the solve)");
            String::new()
        }
    }

    /// Read one command line. Returns `None` on EOF. Uses rustyline when
    /// an editor is active (history / Tab / Ctrl-R); otherwise a plain
    /// reader with a stderr prompt (REPL) or no prompt (JSON).
    fn next_command_line(&mut self) -> Option<String> {
        if let DebugMode::Repl = self.mode {
            if let Some(ed) = self.editor.as_mut() {
                return match ed.readline("pounce-dbg> ") {
                    Ok(l) => {
                        self.prompt_interrupts = 0;
                        let _ = ed.add_history_entry(l.as_str());
                        if let Some(p) = &self.hist_path {
                            let _ = ed.save_history(p);
                        }
                        Some(l)
                    }
                    // Ctrl-C at the prompt: the first cancels the current
                    // line (readline convention); a second in a row quits the
                    // solve, so Ctrl-C is a working escape hatch here too —
                    // matching the running-mode double-tap.
                    Err(ReadlineError::Interrupted) => Some(self.on_prompt_interrupt()),
                    // Ctrl-D / closed input: EOF.
                    Err(ReadlineError::Eof) => None,
                    Err(_) => None,
                };
            }
            let _ = write!(std::io::stderr(), "pounce-dbg> ");
            let _ = std::io::stderr().flush();
            return read_stdin_line();
        }
        // JSON mode reads through the background pump (so async pause can
        // peek the same stream); lazily start it.
        self.pump.get_or_insert_with(StdinPump::start).next()
    }
}

/// Plain blocking line read from stdin; `None` on EOF.
fn read_stdin_line() -> Option<String> {
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => Some(line),
        Err(_) => None,
    }
}

/// Rank residuals by descending magnitude and keep the top `k`.
///
/// Pure (no solver state) so it can be unit-tested directly. Ties on
/// `|value|` keep input order (stable sort), so within equal magnitudes
/// equality constraints precede inequalities precede dual components —
/// the order [`DebugCtx::constraint_residuals`]/`dual_residuals` emit.
/// `k == 0` returns empty.
fn rank_residuals(mut entries: Vec<Residual>, k: usize) -> Vec<Residual> {
    entries.sort_by(|a, b| {
        b.value
            .abs()
            .partial_cmp(&a.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(k);
    entries
}

/// Look up the model name for a residual by kind + split index, given
/// optional split-space names. Equality residuals index the `eq` pool;
/// inequality and `s`-space dual residuals share the `ineq` pool (one
/// slack per inequality); `x`-space dual residuals index `x_var`. Returns
/// `None` when the problem carries no names or the index is out of range.
/// Render a [`RankReport`] into the human-readable REPL lines and the JSON
/// payload for the agent interface. Pure (no solver access) so it can be
/// unit-tested with a synthetic report and a name pool. Shared by the
/// `print rank` command; the `diagnose` finding builds its own one-line
/// summary directly from the report.
fn render_rank_report(
    rep: &RankReport,
    names: &Option<SplitNames>,
    equations: Option<&EquationBook>,
    iter: i32,
) -> (Vec<String>, serde_json::Value) {
    let m = rep.n_rows();
    let n = rep.n_cols;
    let mut lines = vec![
        format!("equality Jacobian J_c: {m} row(s) × {n} column(s)"),
        format!(
            "numerical rank = {} / {}  (deficiency {})",
            rep.rank,
            m,
            rep.deficiency()
        ),
        format!(
            "σ_max = {:.3e}   σ_min = {:.3e}   cond = {}   (rank tol τ = {:.3e})",
            rep.sigma_max(),
            rep.sigma_min(),
            fmt_cond(rep.cond),
            rep.tol
        ),
    ];

    // Singular-value spectrum, capped so a large block stays readable.
    let shown: Vec<String> = rep
        .singular_values
        .iter()
        .take(MAX_SINGULAR_VALUES_SHOWN)
        .map(|s| format!("{s:.3e}"))
        .collect();
    let tail = if rep.singular_values.len() > MAX_SINGULAR_VALUES_SHOWN {
        " …"
    } else {
        ""
    };
    lines.push(format!("singular values: [{}{tail}]", shown.join(", ")));

    if rep.is_rank_deficient() {
        lines.push(format!(
            "rank-deficient: {} equation(s) lie in the near-null space \
             (linearly dependent / redundant) — the source of δ_c regularization:",
            rep.deficiency()
        ));
        let mut shown_any_eq = false;
        for c in rep.culprits.iter().take(MAX_RANK_CULPRITS) {
            let row = &rep.rows[c.row];
            let label = rank_row_label(row, names);
            lines.push(format!("  {label}   (participation {:.2})", c.weight));
            // Print the offending equation's source algebra directly beneath
            // it, so the dependency is readable without a second command.
            // Resolves by model name, so it lands only when the row is named.
            if let Some(eq) = culprit_equation(row, names, equations) {
                lines.push(format!("      {eq}"));
                shown_any_eq = true;
            }
        }
        if rep.culprits.len() > MAX_RANK_CULPRITS {
            lines.push(format!(
                "  … and {} more",
                rep.culprits.len() - MAX_RANK_CULPRITS
            ));
        }
        // Only nag about `print equation` when we couldn't show the algebra
        // inline (no .nl model loaded, or the rows are unnamed).
        if !shown_any_eq {
            lines.push("inspect a row with `print equation <name>` to see its terms".to_string());
        }
    } else {
        lines.push("J_c has full row rank at this iterate.".to_string());
    }

    let culprits_json: Vec<serde_json::Value> = rep
        .culprits
        .iter()
        .map(|c| {
            let row = &rep.rows[c.row];
            serde_json::json!({
                "row": c.row,
                "kind": row.kind.tag(),
                "index": row.index,
                "name": rank_row_name(row, names),
                "label": rank_row_label(row, names),
                "weight": c.weight,
                "equation": culprit_equation(row, names, equations),
            })
        })
        .collect();

    let data = serde_json::json!({
        "iter": iter,
        "n_rows": m,
        "n_cols": n,
        "rank": rep.rank,
        "deficiency": rep.deficiency(),
        "rank_deficient": rep.is_rank_deficient(),
        "sigma_max": rep.sigma_max(),
        "sigma_min": rep.sigma_min(),
        "cond": cond_json(rep.cond),
        "tol": rep.tol,
        "singular_values": rep.singular_values,
        "culprits": culprits_json,
    });

    (lines, data)
}

/// Rendered source algebra of a rank-report culprit row, resolved through
/// the [`EquationBook`] by model name (the same DAG-faithful text `print
/// equation` shows). `None` when no equation book is loaded, the row is
/// unnamed, or the name doesn't resolve — the split equality index the
/// rank report carries is *not* the original `.nl` row index the book keys
/// on, so only named rows can be mapped.
fn culprit_equation(
    row: &RankRow,
    names: &Option<SplitNames>,
    equations: Option<&EquationBook>,
) -> Option<String> {
    let book = equations?;
    let name = rank_row_name(row, names)?;
    let i = book.resolve(&name)?;
    Some(book.equations.get(i)?.clone())
}

/// Model name of a rank-report row, if the problem carries names — the
/// bare name (e.g. `mass_balance`), no `kind[..]` wrapper. `None` when
/// unnamed. Routes through [`resid_name`] so equality/inequality rows hit
/// the same name pools as the rest of the debugger.
fn rank_row_name(row: &RankRow, names: &Option<SplitNames>) -> Option<String> {
    let r = Residual {
        kind: row.kind,
        index: row.index,
        value: 0.0,
    };
    resid_name(&r, names).map(|s| s.to_string())
}

/// Display label for a rank-report row: `c[mass_balance]` when named, else
/// `c[3]` by split index — matching [`worst_named`]'s convention.
fn rank_row_label(row: &RankRow, names: &Option<SplitNames>) -> String {
    match rank_row_name(row, names) {
        Some(name) => format!("{}[{}]", row.kind.tag(), name),
        None => format!("{}[{}]", row.kind.tag(), row.index),
    }
}

/// Human rendering of a condition number, spelling out a non-finite ratio
/// (`σ_min == 0`) as `inf` rather than `NaN`/`inf` float formatting.
fn fmt_cond(cond: f64) -> String {
    if cond.is_finite() {
        format!("{cond:.3e}")
    } else {
        "inf (σ_min = 0)".to_string()
    }
}

/// JSON rendering of a condition number — `null` for a non-finite ratio,
/// since JSON has no infinity.
fn cond_json(cond: f64) -> serde_json::Value {
    if cond.is_finite() {
        serde_json::json!(cond)
    } else {
        serde_json::Value::Null
    }
}

fn resid_name<'a>(r: &Residual, names: &'a Option<SplitNames>) -> Option<&'a str> {
    let n = names.as_ref()?;
    let pool = match r.kind {
        ResidKind::Eq => &n.eq,
        ResidKind::Ineq | ResidKind::DualS => &n.ineq,
        ResidKind::DualX => &n.x_var,
    };
    pool.get(r.index).and_then(|o| o.as_deref())
}

/// The single largest-magnitude residual, labeled with its model name
/// (`c[mass_balance]`) when available, else its split index (`c[3]`),
/// paired with its signed value. `None` for an empty input.
fn worst_named(resids: Vec<Residual>, names: &Option<SplitNames>) -> Option<(String, f64)> {
    let top = rank_residuals(resids, 1);
    let r = top.first()?;
    let label = match resid_name(r, names) {
        Some(name) => format!("{}[{}]", r.kind.tag(), name),
        None => format!("{}[{}]", r.kind.tag(), r.index),
    };
    Some((label, r.value))
}

/// Print the branded open banner (human REPL only): the project POUNCE
/// wordmark (shared with the solve header) over a brief command cheat
/// sheet. Colour only on a TTY and unless `NO_COLOR` is set.
pub fn print_open_banner(mode: DebugMode) {
    if !matches!(mode, DebugMode::Repl) {
        return;
    }
    let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let paint = |r: u8, g: u8, b: u8, bold: bool, s: &str| -> String {
        if color {
            let w = if bold { "1;" } else { "" };
            format!("\x1b[{w}38;2;{r};{g};{b}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    };
    // Project palette: tiger-orange accents, gold highlight, dim text.
    let orange = |s: &str| paint(0xE8, 0x7A, 0x1E, true, s);
    let gold = |s: &str| paint(0xFF, 0xB0, 0x00, true, s);
    let dim = |s: &str| paint(0x7A, 0x7E, 0x88, false, s);
    // One cheat-sheet item: orange key (with shortcut) + dim gloss.
    let item = |key: &str, gloss: &str| format!("{} {}", orange(key), dim(gloss));

    let err = std::io::stderr();
    let mut h = err.lock();
    let _ = writeln!(h);
    // The official wordmark (steel sheen + molten claws), shared with the
    // solve header, rendered to stderr with a small indent.
    for row in crate::print::logo_rows(color) {
        let _ = writeln!(h, "  {row}");
    }
    let _ = writeln!(h);
    let _ = writeln!(
        h,
        "  {}  {}",
        gold("interior-point debugger"),
        dim(&format!(
            "· pounce {} · pdb for the IPM",
            env!("CARGO_PKG_VERSION")
        ))
    );
    let _ = writeln!(h);
    // Most-common commands with their letter shortcuts.
    let _ = writeln!(
        h,
        "  {}   {}   {}   {}   {}",
        item("s", "step"),
        item("c", "continue"),
        item("b", "N break"),
        item("r", "N run"),
        item("q", "quit"),
    );
    let _ = writeln!(
        h,
        "  {}   {}   {}   {}   {}",
        item("p", "x print"),
        item("i", "info"),
        item("set", "x[i] v"),
        item("watch", "x"),
        item("viz", "kkt"),
    );
    let _ = writeln!(
        h,
        "  {} {} {}",
        dim("type"),
        gold("help"),
        dim("for all commands · `ask` to consult Claude · Ctrl-C breaks in"),
    );
    let _ = writeln!(h);
}

/// Whether a command line is an in-band pause request (`pause`, or a JSON
/// `{"cmd":"pause"}`), used for the async-pause-while-running path.
fn is_pause_command(line: &str) -> bool {
    parse_command(line, DebugMode::Json).command.trim() == "pause"
}

/// Background stdin reader for JSON mode. A thread reads newline-delimited
/// commands into a shared queue so the running loop can *peek* for an
/// async `{"cmd":"pause"}` between iterations (no signals — the
/// Windows-friendly path) while the prompt still pops commands blocking.
struct StdinPump {
    inner: std::sync::Arc<(
        std::sync::Mutex<VecDeque<Option<String>>>,
        std::sync::Condvar,
    )>,
}

impl StdinPump {
    fn start() -> Self {
        let inner = std::sync::Arc::new((
            std::sync::Mutex::new(VecDeque::new()),
            std::sync::Condvar::new(),
        ));
        let w = std::sync::Arc::clone(&inner);
        std::thread::spawn(move || {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            let mut lock = stdin.lock();
            let (m, cv) = &*w;
            loop {
                let mut line = String::new();
                let item = match lock.read_line(&mut line) {
                    Ok(0) | Err(_) => None, // EOF / error sentinel
                    Ok(_) => Some(line),
                };
                let done = item.is_none();
                m.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_back(item);
                cv.notify_one();
                if done {
                    break;
                }
            }
        });
        Self { inner }
    }

    /// Blocking pop of the next command line; `None` on EOF (sticky).
    fn next(&self) -> Option<String> {
        let (m, cv) = &*self.inner;
        let mut q = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            match q.front() {
                None => {
                    q = cv
                        .wait(q)
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                }
                Some(None) => return None, // EOF — leave sentinel in place
                Some(Some(_)) => return q.pop_front().flatten(),
            }
        }
    }

    /// Non-blocking: if a queued `pause` request is at the front, consume
    /// it and return true. Leaves any other queued command in place.
    fn try_take_pause(&self) -> bool {
        let (m, _) = &*self.inner;
        let mut q = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(Some(front)) = q.front() {
            if is_pause_command(front) {
                q.pop_front();
                return true;
            }
        }
        false
    }
}

impl DebugHook for SolverDebugger {
    /// Capture the heavy KKT matrix / `LDLᵀ` factor only while attached:
    /// once detached the debugger runs free and won't `viz`, so there's
    /// no reason to pay the O(nnz) assembly every iteration.
    fn wants_kkt_capture(&self) -> bool {
        !self.detached
    }

    /// Re-arm a [`quiet`](SolverDebugger::quiet) debugger to drop in at the
    /// next checkpoint of the next sub-solve (the tree debugger's
    /// step-into-relaxation).
    fn arm(&mut self) {
        self.step = true;
        self.detached = false;
        self.pause_iters = true;
        self.pause_terminal = true;
    }

    fn at_checkpoint(&mut self, ctx: &mut dyn DebugState) -> DebugAction {
        // One-time handshake so a JSON client learns the protocol /
        // capabilities before the first pause.
        if matches!(self.mode, DebugMode::Json) && !self.hello_sent {
            self.emit_hello();
            self.hello_sent = true;
        }
        // Terminal post-mortem checkpoint: pause if configured (and, for
        // `--debug-on-error`, only when the solve failed). Snapshots /
        // rewinding don't apply — the solve is over.
        if let Checkpoint::Terminated = ctx.checkpoint() {
            // An in-flight `sweep`/`multistart` records this solve and
            // launches the next; `Some` means "re-solving from the next
            // seed", `None` means the sweep finished (fall through).
            if self.sweep.is_some() {
                // A sweep can only be started on the NLP solver, so the
                // downcast succeeds whenever one is in flight.
                if let Some(c) = as_nlp(ctx) {
                    if let Some(action) = self.drive_sweep(c) {
                        return action;
                    }
                }
            }
            let failed = ctx.status().map(|s| !is_success_status(s)).unwrap_or(false);
            let should =
                self.pause_terminal && !self.detached && (!self.terminal_only_on_error || failed);
            if !should {
                return DebugAction::Resume;
            }
            self.ensure_editor();
            self.emit_pause(ctx, None);
            return self.prompt_loop(ctx);
        }

        let cp = ctx.checkpoint();
        // Track the restoration bracket so inner-IPM pauses are flagged.
        match cp {
            Checkpoint::PreRestoration => self.in_restoration = true,
            Checkpoint::PostRestoration => self.in_restoration = false,
            _ => {}
        }
        let is_iter_start = matches!(cp, Checkpoint::IterStart);

        // At each iteration top, snapshot the primal-dual state (cheap —
        // Rc clone) so `goto` can reach any seen iteration. Bound memory
        // by evicting the oldest beyond the cap.
        if is_iter_start {
            if let Some(snap) = ctx.snapshot() {
                self.snapshots.insert(ctx.iter(), snap);
                while self.snapshots.len() > SNAPSHOT_CAP {
                    let Some(&oldest) = self.snapshots.keys().next() else {
                        break;
                    };
                    self.snapshots.remove(&oldest);
                }
            }
            // Update μ-stall tracking before events are evaluated.
            self.update_mu_stall(ctx.mu());
        }

        // Decide whether to pause. `stop-at` and a one-shot `stepi` apply
        // at every checkpoint; step / run / breakpoints / conditions /
        // Ctrl-C only at the iteration top.
        let mut reason: Option<String> = None;
        let mut pause = self.sub_step || self.stop_at.contains(cp.as_str());

        // Event breakpoints fire at whatever checkpoint makes them
        // observable (e.g. `regularized` at after_search_dir), so check
        // them at every checkpoint, not just iter_start.
        if let Some(ev) = self.matched_event(ctx) {
            pause = true;
            reason = Some(format!("event: {ev}"));
        }

        if is_iter_start {
            if self.interruptible && interrupt::take() {
                pause = true;
                reason = Some("interrupt (Ctrl-C)".into());
            }
            // In-band async pause: a `{"cmd":"pause"}` that arrived on
            // stdin during the run (JSON mode, #72 §5 option b).
            if let Some(p) = self.pump.as_ref() {
                if p.try_take_pause() {
                    pause = true;
                    reason = Some("pause (requested)".into());
                }
            }
            if self.pause_iters {
                if self.should_pause(ctx.iter()) {
                    pause = true;
                }
                if let Some(c) = self.matched_condition(ctx) {
                    pause = true;
                    reason = Some(c);
                }
            }
            // Watchpoints fire regardless of pause_iters (explicit, like
            // breakpoints); evaluated every iter to keep baselines fresh.
            if let Some(w) = self.matched_watchpoint(ctx) {
                pause = true;
                reason = Some(format!("watchpoint: {w}"));
            }
        }

        if !pause {
            // Not pausing: in JSON mode emit a per-iteration `progress`
            // event (once per outer iter) so a visual debugger isn't blind
            // during a long `continue`. Issue #72 §1.
            if is_iter_start && self.emit_progress && matches!(self.mode, DebugMode::Json) {
                self.emit_progress_event(ctx);
            }
            return DebugAction::Resume;
        }
        // Consume one-shot arming; commands re-arm as needed.
        self.step = false;
        self.sub_step = false;
        self.emit_pause(ctx, reason.as_deref());

        // Auto-run any command list attached to this iteration's
        // breakpoint (`commands N …`). If it resumes/stops, honor that
        // without dropping to the prompt.
        if is_iter_start {
            if let Some(cmds) = self.bp_commands.get(&ctx.iter()).cloned() {
                for c in cmds {
                    let out = self.dispatch(&c, ctx);
                    self.emit_result(&c, &out, None);
                    match out.flow {
                        Flow::Resume => return DebugAction::Resume,
                        Flow::Stop => return DebugAction::Stop,
                        Flow::Stay => {}
                    }
                }
            }
        }

        self.ensure_editor();
        self.prompt_loop(ctx)
    }
}

impl SolverDebugger {
    /// Read and dispatch commands until one resumes or stops the solve.
    fn prompt_loop(&mut self, ctx: &mut dyn DebugState) -> DebugAction {
        // Run a `--debug-script` once, at the first pause, before reading
        // any interactive command. It may itself resume / stop the solve.
        if let Some(path) = self.pending_script.take() {
            let out = self.cmd_source(&[path.as_str()], ctx);
            self.emit_result("source", &out, None);
            match out.flow {
                Flow::Resume => return DebugAction::Resume,
                Flow::Stop => return DebugAction::Stop,
                Flow::Stay => {}
            }
        }
        loop {
            let line = match self.next_command_line() {
                Some(l) => l,
                None => {
                    // EOF on stdin. REPL (Ctrl-D) means "let it run" —
                    // detach and finish, pdb-style. In JSON mode a closed
                    // pipe means the controlling client went away, so
                    // abort the solve rather than run on headless.
                    return match self.mode {
                        DebugMode::Repl => {
                            self.detached = true;
                            DebugAction::Resume
                        }
                        DebugMode::Json => DebugAction::Stop,
                    };
                }
            };
            let parsed = parse_command(&line, self.mode);
            let cmd = parsed.command.trim().to_string();
            if cmd.is_empty() {
                continue;
            }
            let out = self.dispatch(&cmd, ctx);
            self.emit_result(&cmd, &out, parsed.id.as_ref());
            match out.flow {
                Flow::Stay => continue,
                Flow::Resume => return DebugAction::Resume,
                Flow::Stop => return DebugAction::Stop,
            }
        }
    }
}

/// A command read from the input stream: the resolved command string
/// plus an optional client-supplied request id (echoed back as
/// `request_id` so an async client can correlate responses).
struct ParsedCmd {
    command: String,
    id: Option<serde_json::Value>,
}

/// In JSON mode a command line may be a bare string or a JSON object
/// `{"cmd": "...", "args": [...], "id": <any>}`. Returns the resolved
/// command string and the request id (if the object carried one).
fn parse_command(line: &str, mode: DebugMode) -> ParsedCmd {
    let trimmed = line.trim();
    if let DebugMode::Json = mode {
        if trimmed.starts_with('{') {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                let cmd = v.get("cmd").and_then(|c| c.as_str()).unwrap_or("");
                let mut s = cmd.to_string();
                if let Some(args) = v.get("args").and_then(|a| a.as_array()) {
                    for a in args {
                        if let Some(a) = a.as_str() {
                            s.push(' ');
                            s.push_str(a);
                        } else {
                            s.push(' ');
                            s.push_str(&a.to_string());
                        }
                    }
                }
                return ParsedCmd {
                    command: s,
                    id: v.get("id").cloned(),
                };
            }
        }
    }
    ParsedCmd {
        command: trimmed.to_string(),
        id: None,
    }
}

fn emit_json(v: &serde_json::Value) {
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let _ = writeln!(h, "{v}");
    let _ = h.flush();
}

/// Downcast a generic [`DebugState`] to the NLP solver's concrete
/// [`DebugCtx`], for the NLP-only REPL commands (rank diagnosis, model-name
/// resolution, warm `resolve`, sweep/multistart). `None` for the
/// convex/conic and global solvers, whose REPL reports "not supported".
fn as_nlp<'a>(ctx: &'a dyn DebugState) -> Option<&'a DebugCtx> {
    ctx.as_any().and_then(|a| a.downcast_ref::<DebugCtx>())
}

/// Mutable form of [`as_nlp`], for commands that mutate NLP-specific state.
fn as_nlp_mut<'a>(ctx: &'a mut dyn DebugState) -> Option<&'a mut DebugCtx> {
    ctx.as_any_mut().and_then(|a| a.downcast_mut::<DebugCtx>())
}

/// Standard "command needs the NLP solver" error for the convex/global REPL.
fn nlp_only(cmd: &str) -> CmdOut {
    CmdOut::err(format!(
        "`{cmd}` is only available for the NLP solver (not the convex/conic or global solvers)"
    ))
}

/// The iterate-block names the *current* solver exposes (NLP: the eight
/// primal-dual blocks; convex IPM: `x`/`s`/`y`/`z`). Block commands use
/// this rather than the static NLP [`BLOCK_NAMES`] so they work for any
/// solver behind the [`DebugState`] trait.
fn block_names(ctx: &dyn DebugState) -> Vec<&'static str> {
    ctx.block_dims().into_iter().map(|(n, _)| n).collect()
}

/// Whether `name` is one of the current solver's iterate blocks.
fn is_block(ctx: &dyn DebugState, name: &str) -> bool {
    block_names(ctx).iter().any(|n| *n == name)
}

fn fmt_vec(name: &str, v: &[f64]) -> String {
    const MAX: usize = 12;
    if v.len() <= MAX {
        format!(
            "{name} = [{}]",
            v.iter()
                .map(|x| format!("{x:.6e}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        let head = v[..MAX]
            .iter()
            .map(|x| format!("{x:.6e}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{name} = [{head}, … ({} total)]", v.len())
    }
}

fn type_str(t: OptionType) -> &'static str {
    match t {
        OptionType::OT_Number => "Number",
        OptionType::OT_Integer => "Integer",
        OptionType::OT_String => "String",
        OptionType::OT_Unknown => "Unknown",
    }
}

fn default_str(d: &DefaultValue) -> String {
    match d {
        DefaultValue::None => "-".into(),
        DefaultValue::Number(v) => format!("{v}"),
        DefaultValue::Integer(v) => format!("{v}"),
        DefaultValue::String(s) => s.clone(),
    }
}

/// Write `vals` to a temp JSON file and open it in an external viewer.
/// The viewer command comes from `POUNCE_DBG_VIEWER` (a template where
/// `{}` is replaced by the path; if absent, the path is appended), else
/// the platform default (`xdg-open` on Linux, `open` on macOS).
fn write_and_open(label: &str, iter: i32, vals: &[f64]) -> Result<(String, String), String> {
    let payload = serde_json::json!({"label": label, "iter": iter, "values": vals});
    write_json_and_open(label, iter, &payload)
}

/// Build the prompt handed to the LLM by `ask`: a compact, self-contained
/// description of the paused interior-point state plus the user question.
fn build_ask_prompt(ctx: &dyn DebugState, question: &str) -> String {
    use std::fmt::Write as _;
    let mut p = String::new();
    p.push_str(
        "You are helping debug a paused run of POUNCE, a pure-Rust interior-point \
         optimization solver whose NLP core is ported from Ipopt. The solve is \
         stopped at a debugger checkpoint. \
         Use the state below to answer concisely and suggest concrete next steps \
         (options to try, what to inspect). State:\n\n",
    );
    let _ = writeln!(p, "checkpoint = {}", ctx.checkpoint().as_str());
    if let Some(s) = ctx.status() {
        let _ = writeln!(p, "status     = {s}");
    }
    let _ = writeln!(p, "iter       = {}", ctx.iter());
    let _ = writeln!(p, "mu         = {:.6e}", ctx.mu());
    let _ = writeln!(p, "objective  = {:.8e}", ctx.objective());
    let _ = writeln!(p, "inf_pr     = {:.6e}", ctx.inf_pr());
    let _ = writeln!(p, "inf_du     = {:.6e}", ctx.inf_du());
    let _ = writeln!(p, "nlp_error  = {:.6e}", ctx.nlp_error());
    let (ap, ad) = ctx.alpha();
    let _ = writeln!(p, "alpha_pr   = {ap:.4e}, alpha_du = {ad:.4e}");
    let _ = writeln!(p, "ls_trials  = {}", ctx.ls_count());
    let dims: Vec<String> = ctx
        .block_dims()
        .into_iter()
        .map(|(n, d)| format!("{n}:{d}"))
        .collect();
    let _ = writeln!(p, "dims       = {}", dims.join(" "));
    if let Some(k) = ctx.kkt() {
        let _ = writeln!(
            p,
            "kkt        = dim {} inertia n+={} n-={} (expected n-={}, {}) delta_w={:.3e} delta_c={:.3e} status={}",
            k.dim,
            k.n_pos,
            k.n_neg,
            k.expected_neg,
            if k.inertia_correct { "correct" } else { "WRONG" },
            k.delta_w,
            k.delta_c,
            k.status
        );
    }
    let _ = write!(p, "\nQuestion: {question}\n");
    p
}

/// Resolve the LLM command from `$POUNCE_DBG_LLM` (whitespace-split; `{}`
/// substitutes the prompt as an argument), defaulting to `claude -p`. The
/// bool is whether the prompt goes on stdin (true) or was substituted
/// into an argument (false).
fn llm_command(prompt: &str) -> (String, Vec<String>, bool) {
    let tmpl = std::env::var("POUNCE_DBG_LLM").unwrap_or_default();
    let tmpl = tmpl.trim();
    if tmpl.is_empty() {
        return ("claude".to_string(), vec!["-p".to_string()], true);
    }
    let mut parts = tmpl
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let prog = parts.remove(0);
    let mut substituted = false;
    for a in parts.iter_mut() {
        if a.contains("{}") {
            *a = a.replace("{}", prompt);
            substituted = true;
        }
    }
    (prog, parts, !substituted)
}

/// Run the configured LLM command, feeding `prompt` on stdin (unless it
/// was substituted into an argument), and return its stdout.
fn run_llm(prompt: &str) -> Result<String, String> {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let (prog, args, on_stdin) = llm_command(prompt);
    let mut cmd = Command::new(&prog);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd.stdin(if on_stdin {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut child = cmd.spawn().map_err(|e| {
        format!(
            "could not launch `{prog}`: {e} \
             (set POUNCE_DBG_LLM to an LLM command, e.g. `claude -p` or `llm`)"
        )
    })?;
    if on_stdin {
        // Write the prompt and close stdin so the child sees EOF.
        if let Some(mut si) = child.stdin.take() {
            let _ = si.write_all(prompt.as_bytes());
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("`{prog}` failed: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "`{prog}` exited with {}: {}",
            out.status,
            err.trim()
        ));
    }
    let reply = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if reply.is_empty() {
        Err(format!("`{prog}` returned no output"))
    } else {
        Ok(reply)
    }
}

/// Write a JSON artifact to a temp file and open it in an external viewer
/// (`POUNCE_DBG_VIEWER`, else `xdg-open`/`open`). Shared by `viz`.
fn write_json_and_open(
    label: &str,
    iter: i32,
    payload: &serde_json::Value,
) -> Result<(String, String), String> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("pounce-dbg-{label}-iter{iter}.json"));
    std::fs::write(&path, payload.to_string()).map_err(|e| format!("write failed: {e}"))?;
    let path_s = path.to_string_lossy().to_string();

    // Candidate viewers, tried in order until one launches. Each carries
    // the artifact path we report on success (JSON for the data consumers,
    // the rendered HTML for the OS opener):
    //   1. $POUNCE_DBG_VIEWER (a command template; `{}` ← the JSON path),
    //   2. `pounce-dbg-viz` — the bundled interactive Plotly viewer
    //      (`pip install 'pounce-solver[viz]'`), when on PATH,
    //   3. the OS opener (xdg-open / open) on a self-contained HTML
    //      visualization — NOT the raw JSON, which a text editor (VS Code)
    //      would just display instead of plotting.
    let mut candidates: Vec<(String, Vec<String>, String)> = Vec::new();
    match std::env::var("POUNCE_DBG_VIEWER") {
        Ok(tmpl) if !tmpl.trim().is_empty() => {
            let mut parts = tmpl
                .split_whitespace()
                .map(String::from)
                .collect::<Vec<_>>();
            let prog = parts.remove(0);
            let mut replaced = false;
            for a in parts.iter_mut() {
                if a.contains("{}") {
                    *a = a.replace("{}", &path_s);
                    replaced = true;
                }
            }
            if !replaced {
                parts.push(path_s.clone());
            }
            candidates.push((prog, parts, path_s.clone()));
        }
        _ => {
            candidates.push((
                "pounce-dbg-viz".to_string(),
                vec![path_s.clone()],
                path_s.clone(),
            ));
            let opener = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            // Render the HTML spy/bar plot; if that write fails for any
            // reason, fall back to opening the raw JSON.
            let artifact = write_html_viz(label, iter, payload).unwrap_or_else(|_| path_s.clone());
            candidates.push((opener.to_string(), vec![artifact.clone()], artifact));
        }
    }

    let mut last_err = String::new();
    for (program, args, artifact) in &candidates {
        match std::process::Command::new(program).args(args).spawn() {
            Ok(_) => return Ok((artifact.clone(), format!("{program} {}", args.join(" ")))),
            Err(e) => last_err = format!("`{program}`: {e}"),
        }
    }
    Err(format!(
        "wrote {path_s} but could not launch a viewer ({last_err}). \
         Install the interactive viewer (`pip install 'pounce-solver[viz]'`) \
         or set POUNCE_DBG_VIEWER, e.g. `python my_plot.py {{}}`."
    ))
}

/// Render a self-contained HTML visualization (no external assets, no pip
/// install) for a `viz` payload and write it next to the JSON. A KKT/L
/// matrix becomes a sign-colored sparsity (spy) plot; a plain vector
/// becomes a zero-centered bar chart. Opening this in the OS default
/// handler pops a browser window that actually draws the artifact —
/// unlike the raw JSON, which a text editor (VS Code) would just display.
fn write_html_viz(label: &str, iter: i32, payload: &serde_json::Value) -> Result<String, String> {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("pounce-dbg-{label}-iter{iter}.html"));
    let html = VIZ_HTML_TEMPLATE.replace("__PAYLOAD__", &payload.to_string());
    std::fs::write(&path, html).map_err(|e| format!("write failed: {e}"))?;
    Ok(path.to_string_lossy().to_string())
}

/// Self-contained HTML viewer for `viz` artifacts. `__PAYLOAD__` is
/// replaced with the JSON payload; an inline canvas renderer picks the
/// plot type from the payload shape (`matrix` → KKT spy, `l_irn` → L-factor
/// spy, `values` → vector bar chart).
const VIZ_HTML_TEMPLATE: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>pounce-dbg viz</title>
<style>
 html,body{margin:0;background:#0e1116;color:#d6dae0;
   font:13px/1.5 -apple-system,BlinkMacSystemFont,"SF Mono",Menlo,monospace}
 .wrap{padding:18px 20px;max-width:880px;margin:0 auto}
 h1{font-size:15px;margin:0 0 4px;font-weight:600}
 .sub{color:#7d8694;margin:0 0 12px}
 .stats{color:#9aa4b2;white-space:pre-wrap;margin:0 0 14px;
   background:#161b22;border:1px solid #21262d;border-radius:6px;padding:10px 12px}
 canvas{background:#161b22;border:1px solid #30363d;border-radius:6px;
   max-width:100%;height:auto;image-rendering:pixelated}
 .legend{margin-top:10px;color:#9aa4b2}
 .pos{color:#4ea1ff}.neg{color:#ff6b6b}.bad{color:#ff6b6b;font-weight:600}
 .ok{color:#56d364;font-weight:600}
</style></head><body><div class="wrap">
<h1 id="title">pounce-dbg</h1>
<div class="sub" id="sub"></div>
<div class="stats" id="stats"></div>
<canvas id="c" width="820" height="820"></canvas>
<div class="legend" id="legend"></div>
</div>
<script>
const D = __PAYLOAD__;
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');
const $ = id => document.getElementById(id);
const fmt = x => (x===null||x===undefined) ? '—'
  : (Math.abs(x) >= 1e4 || (x!==0 && Math.abs(x) < 1e-3) ? x.toExponential(3) : (+x).toPrecision(6));

function clearCanvas(){ ctx.fillStyle='#161b22'; ctx.fillRect(0,0,cv.width,cv.height); }

function spy(irn, jcn, vals, dim, symmetric, title){
  $('sub').textContent = title;
  clearCanvas();
  const W=cv.width, H=cv.height, pad=42;
  const span=Math.max(1, dim);
  const cell=(Math.min(W,H)-2*pad)/span;
  const px=Math.max(0.7, cell);
  // frame + light grid ticks
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.strokeRect(pad-0.5, pad-0.5, span*cell+1, span*cell+1);
  ctx.fillStyle='#6e7681'; ctx.font='11px monospace';
  ctx.fillText('0', pad-12, pad+9);
  ctx.fillText(String(dim), pad+span*cell-8, pad-8);
  ctx.fillText('row', pad-34, pad+span*cell/2);
  ctx.fillText('col', pad+span*cell/2-8, pad-22);
  let nnz=0;
  for(let k=0;k<irn.length;k++){
    const i=irn[k]-1, j=jcn[k]-1, v=vals?vals[k]:1;
    ctx.fillStyle = v>=0 ? 'rgba(78,161,255,0.92)' : 'rgba(255,107,107,0.92)';
    ctx.fillRect(pad+j*cell, pad+i*cell, px, px); nnz++;
    if(symmetric && i!==j){ ctx.fillRect(pad+i*cell, pad+j*cell, px, px); nnz++; }
  }
  $('legend').innerHTML =
    `<span class="pos">■</span> positive&nbsp;&nbsp;<span class="neg">■</span> negative`
    + `&nbsp;&nbsp;·&nbsp;&nbsp;${dim}×${dim}, ${nnz} plotted nonzeros`
    + (symmetric ? ' (lower triangle mirrored)' : '');
}

function bars(values, title){
  $('sub').textContent = title;
  clearCanvas();
  const W=cv.width, H=cv.height, pad=42;
  const n=values.length;
  const maxAbs=Math.max(1e-300, ...values.map(v=>Math.abs(v)));
  const x0=pad, y0=H-pad, plotW=W-2*pad, plotH=H-2*pad, mid=pad+plotH/2;
  const bw=Math.max(0.7, plotW/Math.max(1,n));
  // zero axis
  ctx.strokeStyle='#30363d'; ctx.beginPath();
  ctx.moveTo(pad, mid); ctx.lineTo(W-pad, mid); ctx.stroke();
  ctx.fillStyle='#6e7681'; ctx.font='11px monospace';
  ctx.fillText('+'+fmt(maxAbs), 4, pad+10);
  ctx.fillText('-'+fmt(maxAbs), 4, H-pad-2);
  ctx.fillText('0', 4, mid+4);
  for(let k=0;k<n;k++){
    const v=values[k], h=(Math.abs(v)/maxAbs)*(plotH/2);
    ctx.fillStyle = v>=0 ? 'rgba(78,161,255,0.92)' : 'rgba(255,107,107,0.92)';
    if(v>=0) ctx.fillRect(pad+k*bw, mid-h, bw, h);
    else     ctx.fillRect(pad+k*bw, mid, bw, h);
  }
  $('legend').innerHTML = `${n} components · max |val| = ${fmt(maxAbs)}`;
}

const lbl = D.label || 'viz';
const iter = (D.iter!==undefined) ? D.iter : '?';
$('title').textContent = `pounce-dbg · viz ${lbl} · iter ${iter}`;

if(D.matrix && D.matrix.irn){
  const m=D.matrix;
  const inertia = (D.inertia_correct===false)
    ? `<span class="bad">WRONG</span>` : `<span class="ok">correct</span>`;
  $('stats').innerHTML =
    `KKT augmented system   dim=${D.dim}\n`+
    `inertia  n+=${D.n_pos}  n-=${D.n_neg}  (expected n-=${D.expected_neg}, ${inertia})\n`+
    `regularization  delta_w=${fmt(D.delta_w)}  delta_c=${fmt(D.delta_c)}\n`+
    `factorization status: ${D.status}`;
  spy(m.irn, m.jcn, m.vals, m.dim, true, 'sparsity pattern (sign-colored)');
} else if(D.l_irn){
  $('stats').textContent =
    `LDLᵀ factor   n=${D.n}   nnz(L)=${D.l_irn.length}   format=${D.format||''}`;
  spy(D.l_irn, D.l_jcn, D.l_vals, D.n, false, 'L factor sparsity (permuted, strict lower)');
} else if(D.values){
  $('stats').textContent = `vector ${lbl}   length=${D.values.length}`;
  bars(D.values, 'component magnitudes (zero-centered)');
} else {
  $('stats').textContent = 'unrecognized payload — raw JSON:\n'+JSON.stringify(D,null,2);
}
</script></body></html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    fn dbg(mode: DebugMode) -> SolverDebugger {
        SolverDebugger::new(mode, None)
    }

    #[test]
    fn json_command_object_is_flattened() {
        assert_eq!(
            parse_command("{\"cmd\":\"print x\"}", DebugMode::Json).command,
            "print x"
        );
        let p = parse_command(
            "{\"cmd\":\"set\",\"args\":[\"x[0]\",\"1.5\"],\"id\":7}",
            DebugMode::Json,
        );
        assert_eq!(p.command, "set x[0] 1.5");
        // Request id is captured for response correlation.
        assert_eq!(p.id, Some(serde_json::json!(7)));
        // Bare strings pass through in either mode, with no id.
        let s = parse_command("step\n", DebugMode::Json);
        assert_eq!(s.command, "step");
        assert!(s.id.is_none());
        assert_eq!(
            parse_command("  print x \n", DebugMode::Repl).command,
            "print x"
        );
    }

    #[test]
    fn pauses_at_first_checkpoint_then_only_when_rearmed() {
        let mut d = dbg(DebugMode::Repl);
        // Fresh debugger is armed (step=true) so it pauses at iter 0.
        assert!(d.should_pause(0));
        // After consuming the arming (as at_checkpoint does), no pause.
        d.step = false;
        assert!(!d.should_pause(1));
        assert!(!d.should_pause(2));
    }

    #[test]
    fn breakpoints_and_run_to_arm_pauses() {
        let mut d = dbg(DebugMode::Repl);
        d.step = false;
        d.breaks = vec![3, 7];
        assert!(!d.should_pause(2));
        assert!(d.should_pause(3));
        assert!(d.should_pause(7));
        // run_to fires once at/after target, then disarms.
        d.run_to = Some(5);
        assert!(!d.should_pause(4));
        assert!(d.should_pause(5));
        assert_eq!(d.run_to, None);
        assert!(!d.should_pause(6));
    }

    #[test]
    fn atom_parses_metric_op_threshold() {
        let a = Atom::parse("mu<1e-4").unwrap();
        assert_eq!(a.metric, Metric::Mu);
        assert_eq!(a.op, CmpOp::Lt);
        assert_eq!(a.rhs, 1e-4);

        // `<=` must not be truncated to `<`.
        let a = Atom::parse("inf_pr<=1e-6").unwrap();
        assert_eq!(a.metric, Metric::InfPr);
        assert_eq!(a.op, CmpOp::Le);

        let a = Atom::parse("iter==10").unwrap();
        assert_eq!(a.metric, Metric::Iter);
        assert_eq!(a.op, CmpOp::Eq);
        assert_eq!(a.rhs, 10.0);
    }

    #[test]
    fn atom_parse_rejects_garbage() {
        assert!(Atom::parse("inf_pr 1e-6").is_err()); // no operator
        assert!(Atom::parse("bogus<1").is_err()); // unknown metric
        assert!(Atom::parse("mu<abc").is_err()); // bad threshold
    }

    #[test]
    fn compound_condition_parses_and_evaluates_left_to_right() {
        // Chain length + joins.
        let c = Condition::parse("mu<1e-4&&inf_pr>1e-3").unwrap();
        assert_eq!(c.rest.len(), 1);
        assert_eq!(c.rest[0].0, Join::And);

        // Parens are stripped; `||` recognized.
        let c = Condition::parse("iter>10&&(inf_du>1e-2||obj<0)").unwrap();
        assert_eq!(c.rest.len(), 2);
        assert_eq!(c.rest[0].0, Join::And);
        assert_eq!(c.rest[1].0, Join::Or);
        assert_eq!(c.raw, "iter>10&&inf_du>1e-2||obj<0");

        // A bad atom anywhere fails the whole parse.
        assert!(Condition::parse("mu<1e-4&&bogus>0").is_err());
    }

    #[test]
    fn completion_is_context_sensitive() {
        // First token completes command verbs.
        let c = completion_candidates(None, "", "co");
        assert!(c.contains(&"continue".to_string()));
        assert!(c.contains(&"complete".to_string()));
        assert!(!c.contains(&"step".to_string()));

        // After `set`, both mu/opt and block names are offered.
        let c = completion_candidates(None, "set ", "");
        assert!(c.contains(&"mu".to_string()));
        assert!(c.contains(&"opt".to_string()));
        assert!(c.contains(&"x".to_string()));

        // After `break if`, metric names.
        let c = completion_candidates(None, "break if ", "inf");
        assert!(c.contains(&"inf_pr".to_string()));
        assert!(c.contains(&"inf_du".to_string()));
        assert!(!c.contains(&"mu".to_string()));

        // `print` completes blocks + scalar keywords.
        let c = completion_candidates(None, "print ", "");
        assert!(c.contains(&"x".to_string()));
        assert!(c.contains(&"obj".to_string()));
    }

    #[test]
    fn cmp_op_truth_table() {
        assert!(CmpOp::Lt.eval(1.0, 2.0));
        assert!(!CmpOp::Lt.eval(2.0, 2.0));
        assert!(CmpOp::Le.eval(2.0, 2.0));
        assert!(CmpOp::Gt.eval(3.0, 2.0));
        assert!(CmpOp::Ge.eval(2.0, 2.0));
        assert!(CmpOp::Eq.eval(2.0, 2.0));
        assert!(!CmpOp::Eq.eval(2.0, 2.5));
    }

    #[test]
    fn interrupt_is_consumed_once() {
        interrupt::set_pending_for_test();
        assert!(interrupt::take(), "first take sees the pending Ctrl-C");
        assert!(!interrupt::take(), "second take is clear (consumed once)");
    }

    #[test]
    fn on_interrupt_constructor_runs_free_but_interruptible() {
        let d = SolverDebugger::on_interrupt(DebugMode::Repl, None);
        assert!(!d.pause_iters, "on-interrupt does not pause each iter");
        assert!(!d.pause_terminal, "on-interrupt does not pause at terminal");
        assert!(d.interruptible, "on-interrupt honors Ctrl-C");
        assert!(!d.step, "on-interrupt starts un-armed");
    }

    #[test]
    fn coffee_easter_egg_prints_art_but_stays_hidden() {
        let d = SolverDebugger::new(DebugMode::Repl, None);
        let out = d.cmd_coffee();
        assert!(out.ok);
        assert!(out.lines.len() > 5, "multi-line art");
        assert!(
            out.lines.iter().any(|l| l.contains("COFFEE")),
            "the mug says COFFEE"
        );
        // Easter egg: not advertised anywhere discoverable.
        assert!(
            !COMMANDS.contains(&"coffee"),
            "hidden from help/complete/Tab"
        );
        // Output is plain in the (non-TTY) test context — no escape codes.
        assert!(
            out.lines.iter().all(|l| !l.contains('\x1b')),
            "no color when stderr isn't a TTY"
        );
    }

    #[test]
    fn double_ctrl_c_at_prompt_quits_single_cancels_line() {
        let mut d = SolverDebugger::new(DebugMode::Repl, None);
        // First Ctrl-C in a row cancels the line (empty → reprompt).
        assert_eq!(d.on_prompt_interrupt(), "");
        // Second in a row quits the solve.
        assert_eq!(d.on_prompt_interrupt(), "quit");
        // Counter reset after quitting, so the next single press cancels again.
        assert_eq!(d.on_prompt_interrupt(), "");
        // A real command in between resets the streak (simulating the
        // `Ok(l)` branch of `next_command_line`).
        d.prompt_interrupts = 0;
        assert_eq!(d.on_prompt_interrupt(), "", "fresh streak after a command");
    }

    #[test]
    fn stop_at_accepts_names_and_aliases() {
        let mut d = SolverDebugger::new(DebugMode::Repl, None);
        assert!(d.cmd_stop_at(&["after_search_dir"]).ok);
        assert!(d.stop_at.contains("after_search_dir"));
        // Aliases canonicalize.
        assert!(d.cmd_stop_at(&["mu"]).ok);
        assert!(d.stop_at.contains("after_mu"));
        assert!(d.cmd_stop_at(&["kkt"]).ok);
        assert!(d.stop_at.contains("after_search_dir"));
        // Unknown name is rejected.
        assert!(!d.cmd_stop_at(&["bogus"]).ok);
        // Clear empties the set.
        assert!(d.cmd_stop_at(&["clear"]).ok);
        assert!(d.stop_at.is_empty());
    }

    #[test]
    fn llm_command_defaults_and_overrides() {
        // Default is `claude -p`, prompt on stdin.
        std::env::remove_var("POUNCE_DBG_LLM");
        let (prog, args, on_stdin) = llm_command("hi");
        assert_eq!(prog, "claude");
        assert_eq!(args, vec!["-p".to_string()]);
        assert!(on_stdin);

        // `{}` substitution puts the prompt in an arg (no stdin).
        std::env::set_var("POUNCE_DBG_LLM", "mytool --ask {}");
        let (prog, args, on_stdin) = llm_command("why");
        assert_eq!(prog, "mytool");
        assert_eq!(args, vec!["--ask".to_string(), "why".to_string()]);
        assert!(!on_stdin);

        // No `{}` ⇒ prompt on stdin.
        std::env::set_var("POUNCE_DBG_LLM", "llm -m gpt");
        let (_, _, on_stdin) = llm_command("q");
        assert!(on_stdin);
        std::env::remove_var("POUNCE_DBG_LLM");
    }

    #[test]
    fn detach_disables_all_pausing() {
        let mut d = dbg(DebugMode::Repl);
        d.detached = true;
        d.step = true;
        d.breaks = vec![1];
        assert!(!d.should_pause(0));
        assert!(!d.should_pause(1));
    }

    #[test]
    fn kkt_capture_tracks_attached_state() {
        // Heavy KKT/L capture is on while stepping (attached), off once
        // detached so a free run doesn't pay the per-iteration assembly.
        let mut d = dbg(DebugMode::Repl);
        assert!(d.wants_kkt_capture());
        d.detached = true;
        assert!(!d.wants_kkt_capture());
    }

    fn resid(kind: ResidKind, index: usize, value: f64) -> Residual {
        Residual { kind, index, value }
    }

    #[test]
    fn rank_residuals_sorts_by_magnitude_and_truncates() {
        use ResidKind::*;
        let entries = vec![
            resid(Eq, 0, -0.5),
            resid(Ineq, 1, 3.0),
            resid(DualX, 2, -7.0),
            resid(DualS, 3, 1.0),
        ];
        let top = rank_residuals(entries, 2);
        assert_eq!(top.len(), 2);
        // Largest |value| first: |-7|, then |3|.
        assert_eq!(top[0].value, -7.0);
        assert_eq!(top[0].kind, DualX);
        assert_eq!(top[1].value, 3.0);
        assert_eq!(top[1].kind, Ineq);
    }

    #[test]
    fn rank_residuals_k_zero_and_k_over_len() {
        use ResidKind::*;
        let entries = vec![resid(Eq, 0, 1.0), resid(Ineq, 1, 2.0)];
        assert!(rank_residuals(entries.clone(), 0).is_empty());
        // k larger than the input just returns everything, ranked.
        let all = rank_residuals(entries, 99);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].value, 2.0);
    }

    #[test]
    fn rank_residuals_is_stable_on_magnitude_ties() {
        use ResidKind::*;
        // Equal |value|: input order preserved (Eq before Ineq before dual).
        let entries = vec![
            resid(Ineq, 5, -2.0),
            resid(Eq, 1, 2.0),
            resid(DualX, 9, -2.0),
        ];
        let top = rank_residuals(entries, 3);
        assert_eq!(
            top.iter().map(|r| r.kind).collect::<Vec<_>>(),
            vec![Ineq, Eq, DualX]
        );
    }

    fn split_names_fixture() -> SplitNames {
        SplitNames {
            x_var: vec![Some("T_reactor".into()), None],
            eq: vec![Some("mass_balance".into()), Some("energy_balance".into())],
            ineq: vec![Some("pressure_cap".into())],
        }
    }

    #[test]
    fn resid_name_maps_each_kind_to_its_pool() {
        use ResidKind::*;
        let names = Some(split_names_fixture());
        // Equality → eq pool; inequality and s-space dual → ineq pool;
        // x-space dual → x_var pool.
        assert_eq!(
            resid_name(&resid(Eq, 1, 0.0), &names),
            Some("energy_balance")
        );
        assert_eq!(
            resid_name(&resid(Ineq, 0, 0.0), &names),
            Some("pressure_cap")
        );
        assert_eq!(
            resid_name(&resid(DualS, 0, 0.0), &names),
            Some("pressure_cap")
        );
        assert_eq!(resid_name(&resid(DualX, 0, 0.0), &names), Some("T_reactor"));
        // Unnamed slot (None) and out-of-range fall back to no name.
        assert_eq!(resid_name(&resid(DualX, 1, 0.0), &names), None);
        assert_eq!(resid_name(&resid(Eq, 9, 0.0), &names), None);
        // No names at all ⇒ None.
        assert_eq!(resid_name(&resid(Eq, 0, 0.0), &None), None);
    }

    #[test]
    fn worst_named_picks_largest_and_labels_it() {
        use ResidKind::*;
        let names = Some(split_names_fixture());
        // |−3.2| is the largest; it sits in the eq pool at index 1.
        let resids = vec![resid(Eq, 0, 0.5), resid(Eq, 1, -3.2), resid(Ineq, 0, 1.1)];
        assert_eq!(
            worst_named(resids, &names),
            Some(("c[energy_balance]".to_string(), -3.2))
        );
        // Without names, the label falls back to the split index.
        let resids = vec![resid(DualX, 7, 9.0)];
        assert_eq!(
            worst_named(resids, &None),
            Some(("grad_x_L[7]".to_string(), 9.0))
        );
        // Empty input ⇒ None.
        assert_eq!(worst_named(vec![], &names), None);
    }

    use pounce_algorithm::debug_rank::RankCulprit;

    fn rank_report_fixture() -> RankReport {
        // 2×3 equality block, row 1 redundant: rank 1, deficiency 1, with
        // both equality rows sharing the single null direction.
        RankReport {
            rows: vec![
                RankRow {
                    kind: ResidKind::Eq,
                    index: 0,
                },
                RankRow {
                    kind: ResidKind::Eq,
                    index: 1,
                },
            ],
            n_cols: 3,
            singular_values: vec![2.0, 0.0],
            tol: 1e-15,
            rank: 1,
            cond: f64::INFINITY,
            culprits: vec![
                RankCulprit {
                    row: 0,
                    weight: 0.5,
                },
                RankCulprit {
                    row: 1,
                    weight: 0.5,
                },
            ],
        }
    }

    #[test]
    fn render_rank_report_names_culprits_and_builds_json() {
        let names = Some(split_names_fixture());
        let rep = rank_report_fixture();
        // No equation book ⇒ names only, plus the `print equation` hint.
        let (lines, data) = render_rank_report(&rep, &names, None, 7);

        let text = lines.join("\n");
        assert!(text.contains("2 row(s) × 3 column(s)"), "{text}");
        assert!(text.contains("numerical rank = 1 / 2"), "{text}");
        // cond is non-finite (σ_min = 0) ⇒ spelled out, not "inf"/"NaN".
        assert!(text.contains("inf (σ_min = 0)"), "{text}");
        // Culprits resolved to model names from the eq pool.
        assert!(text.contains("c[mass_balance]"), "{text}");
        assert!(text.contains("c[energy_balance]"), "{text}");
        assert!(text.contains("participation 0.50"), "{text}");
        // No book ⇒ fall back to the inspect hint, no inline algebra.
        assert!(text.contains("print equation"), "{text}");

        // JSON payload: cond is null (non-finite), culprits carry names but
        // no resolved equation (no book).
        assert_eq!(data["iter"], 7);
        assert_eq!(data["rank"], 1);
        assert_eq!(data["deficiency"], 1);
        assert_eq!(data["rank_deficient"], true);
        assert!(data["cond"].is_null(), "non-finite cond ⇒ null: {data}");
        assert_eq!(data["culprits"][0]["name"], "mass_balance");
        assert_eq!(data["culprits"][0]["label"], "c[mass_balance]");
        assert!(data["culprits"][0]["equation"].is_null());
        assert_eq!(data["culprits"][1]["name"], "energy_balance");
    }

    #[test]
    fn render_rank_report_prints_culprit_equations_inline() {
        let names = Some(split_names_fixture());
        let rep = rank_report_fixture();
        // The equation book keys on original .nl row order; both eq names
        // present so the rank culprits resolve by name.
        let book = EquationBook::new(
            vec!["mass_balance".into(), "energy_balance".into()],
            vec![
                "x[0] + x[1] - 10 = 0".into(),
                "T_reactor*flow - Q = 0".into(),
            ],
        );
        let (lines, data) = render_rank_report(&rep, &names, Some(&book), 7);

        let text = lines.join("\n");
        // The offending equations' algebra is printed inline, beneath each
        // named culprit — no second command needed.
        assert!(text.contains("x[0] + x[1] - 10 = 0"), "{text}");
        assert!(text.contains("T_reactor*flow - Q = 0"), "{text}");
        // With the algebra shown inline, the `print equation` nag is dropped.
        assert!(!text.contains("inspect a row with"), "{text}");

        // JSON carries the resolved equation per culprit.
        assert_eq!(data["culprits"][0]["equation"], "x[0] + x[1] - 10 = 0");
        assert_eq!(data["culprits"][1]["equation"], "T_reactor*flow - Q = 0");
    }

    #[test]
    fn render_rank_report_full_rank_reports_positive_signal() {
        let rep = RankReport {
            rows: vec![
                RankRow {
                    kind: ResidKind::Eq,
                    index: 0,
                },
                RankRow {
                    kind: ResidKind::Eq,
                    index: 1,
                },
            ],
            n_cols: 3,
            singular_values: vec![2.0, 1.0],
            tol: 1e-15,
            rank: 2,
            cond: 2.0,
            culprits: vec![],
        };
        let (lines, data) = render_rank_report(&rep, &None, None, 3);
        let text = lines.join("\n");
        assert!(text.contains("full row rank"), "{text}");
        assert!(!text.contains("rank-deficient"), "{text}");
        assert_eq!(data["rank_deficient"], false);
        assert_eq!(data["cond"], 2.0);
        assert_eq!(data["culprits"].as_array().map(|a| a.len()), Some(0));
    }

    #[test]
    fn print_equation_resolves_by_name_index_and_errors() {
        let mut d = dbg(DebugMode::Repl);
        // No book wired in yet ⇒ a helpful error, not a panic.
        let out = d.cmd_print_equation(&[]);
        assert!(!out.ok);
        assert!(out.lines[0].contains("needs an .nl model"));

        d.set_equation_book(EquationBook::new(
            vec!["mass_balance".into(), String::new()],
            vec!["x[0] + x[1] = 10".into(), "x[0] - x[1] <= 2".into()],
        ));

        // No arg ⇒ count + usage hint.
        let out = d.cmd_print_equation(&[]);
        assert!(out.ok);
        assert!(out.lines[0].contains("2 constraint equation"));

        // By model name.
        let out = d.cmd_print_equation(&["mass_balance"]);
        assert!(out.ok);
        assert_eq!(out.lines[0], "mass_balance:  x[0] + x[1] = 10");

        // By original row index; the unnamed row falls back to `c[1]`.
        let out = d.cmd_print_equation(&["1"]);
        assert!(out.ok);
        assert_eq!(out.lines[0], "c[1]:  x[0] - x[1] <= 2");

        // Unknown key ⇒ error.
        let out = d.cmd_print_equation(&["nope"]);
        assert!(!out.ok);
        assert!(out.lines[0].contains("no constraint named or indexed"));
    }

    /// Build an `EqualityIncidence` from an explicit row→vars adjacency,
    /// carrying the original-row indices so `con_label`'s `c[orig]`
    /// fallback can be exercised.
    fn eq_inc(n_vars: usize, eq_row_inner_idx: Vec<usize>, rows: &[&[usize]]) -> EqualityIncidence {
        let mut adj_ptr = vec![0usize];
        let mut vars = Vec::new();
        for r in rows {
            let mut v = r.to_vec();
            v.sort_unstable();
            v.dedup();
            vars.extend_from_slice(&v);
            adj_ptr.push(vars.len());
        }
        EqualityIncidence {
            n_vars,
            eq_row_inner_idx,
            adj_ptr,
            vars,
        }
    }

    #[test]
    fn structural_singularity_names_overdetermined_equations() {
        // 3 equality rows over 2 vars, each touching both → a maximum
        // matching saturates the 2 columns, leaving 1 row unmatched;
        // the alternating walk pulls all 3 rows into the over-determined
        // block. The finding must name every candidate equation, the
        // shared variables, and the ≥1 redundancy excess.
        let inc = eq_inc(2, vec![0, 1, 2], &[&[0, 1], &[0, 1], &[0, 1]]);
        let book = StructureBook::new(
            inc,
            vec!["balance_a".into(), "balance_b".into(), "balance_c".into()],
            vec!["flow".into(), "temp".into()],
        );
        let f = book.findings();
        assert_eq!(f.len(), 1);
        let (sev, code, msg) = &f[0];
        assert_eq!(*sev, "warning");
        assert_eq!(*code, "structural_singularity");
        assert!(msg.contains("balance_a"), "msg: {msg}");
        assert!(msg.contains("balance_b"), "msg: {msg}");
        assert!(msg.contains("balance_c"), "msg: {msg}");
        assert!(msg.contains("flow") && msg.contains("temp"), "msg: {msg}");
        assert!(msg.contains("≥1"), "msg: {msg}");
    }

    #[test]
    fn structural_findings_silent_when_well_posed_and_fall_back_to_indices() {
        // Square 2×2 with a perfect matching → structurally sound, no
        // finding (and the normal "more vars than eqs" case is never
        // flagged either, since we only report the over-determined side).
        let inc = eq_inc(2, vec![0, 1], &[&[0], &[1]]);
        let book = StructureBook::new(inc, vec![], vec![]);
        assert!(book.findings().is_empty());

        // Over-determined but unnamed: 3 rows over 1 var, with the
        // original row indices skipping 2 (e.g. an interleaved
        // inequality) → labels fall back to `c[<orig>]`.
        let inc = eq_inc(1, vec![0, 1, 3], &[&[0], &[0], &[0]]);
        let book = StructureBook::new(inc, vec![], vec![]);
        let f = book.findings();
        assert_eq!(f.len(), 1);
        let msg = &f[0].2;
        assert!(
            msg.contains("c[0]") && msg.contains("c[1]") && msg.contains("c[3]"),
            "msg: {msg}"
        );
    }

    #[test]
    fn structural_singularity_handles_empty_row_with_no_variables() {
        // An empty equality row (no variable support) is unmatched and
        // touches no columns → over-determined with no shared variables.
        let inc = eq_inc(1, vec![0, 1], &[&[0], &[]]);
        let book = StructureBook::new(inc, vec!["real".into(), "ghost".into()], vec!["x".into()]);
        let f = book.findings();
        assert_eq!(f.len(), 1);
        let msg = &f[0].2;
        assert!(msg.contains("ghost"), "msg: {msg}");
        assert!(msg.contains("no variables"), "msg: {msg}");
    }

    #[test]
    fn parse_floats_accepts_commas_whitespace_and_newlines() {
        assert_eq!(parse_floats("1, 2 ,3").unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(parse_floats("1\n2\n-3.5").unwrap(), vec![1.0, 2.0, -3.5]);
        assert_eq!(parse_floats("  1.0   2e-1 ").unwrap(), vec![1.0, 0.2]);
        assert!(parse_floats("1, nope, 3").is_err());
        assert_eq!(parse_floats("").unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn jitter_start_zero_is_the_unperturbed_base_and_is_deterministic() {
        let base = vec![1.0, -2.0, 0.0];
        // k=0 reproduces the base exactly, so a multistart always covers x0.
        assert_eq!(jitter(&base, 0.1, 0), base);
        // k>0 perturbs, bounded by rel·(|xᵢ|+1), and reproduces run-to-run.
        let a = jitter(&base, 0.1, 1);
        let b = jitter(&base, 0.1, 1);
        assert_eq!(a, b);
        assert_ne!(a, base);
        for (j, (&p, &x)) in a.iter().zip(&base).enumerate() {
            let bound = 0.1 * (x.abs() + 1.0);
            assert!(
                (p - x).abs() <= bound + 1e-12,
                "component {j} moved {} > bound {bound}",
                (p - x).abs()
            );
        }
        // Different start index → different point.
        assert_ne!(jitter(&base, 0.1, 1), jitter(&base, 0.1, 2));
    }

    #[test]
    fn sample_start_draws_inside_finite_boxes_and_jitters_unbounded() {
        let base = vec![1.0, 1.0, 0.5];
        // var 0: box [0,2]; var 1: lower-only (upper = +inf); var 2: box [-1,1].
        let lo = vec![0.0, 0.0, -1.0];
        let hi = vec![2.0, f64::INFINITY, 1.0];
        let b = Some((lo.as_slice(), hi.as_slice()));
        // Start 0 is always the base, regardless of bounds.
        assert_eq!(sample_start(&base, b, 0.1, 0), base);
        for k in 1..50 {
            let s = sample_start(&base, b, 0.1, k);
            // Boxed components land strictly inside their box.
            assert!((0.0..=2.0).contains(&s[0]), "var0 {} out of [0,2]", s[0]);
            assert!((-1.0..=1.0).contains(&s[2]), "var2 {} out of [-1,1]", s[2]);
            // The half-bounded component falls back to jitter around base.
            let bound = 0.1 * (base[1].abs() + 1.0);
            assert!(
                (s[1] - base[1]).abs() <= bound + 1e-12,
                "var1 jitter exceeded"
            );
        }
        // Deterministic in k.
        assert_eq!(
            sample_start(&base, b, 0.1, 7),
            sample_start(&base, b, 0.1, 7)
        );
    }

    #[test]
    fn path_completion_lists_matching_files_with_dir_prefix() {
        let dir = std::env::temp_dir().join("pounce_dbg_complete_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("starts.txt"), "0,0\n").unwrap();
        std::fs::write(dir.join("start2.txt"), "1,1\n").unwrap();
        std::fs::write(dir.join("other.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.join("subdir")).unwrap();

        let p = dir.to_string_lossy().to_string();
        // Prefix filters; the dir prefix is preserved so the token replaces whole.
        let mut got = path_candidates(&format!("{p}/start"));
        got.sort();
        assert_eq!(
            got,
            vec![format!("{p}/start2.txt"), format!("{p}/starts.txt")]
        );
        // Directories get a trailing slash.
        let got = path_candidates(&format!("{p}/sub"));
        assert_eq!(got, vec![format!("{p}/subdir/")]);
        // Listing a directory with an empty basename returns all entries.
        assert_eq!(path_candidates(&format!("{p}/")).len(), 4);
        // Verb-context routing: `load <file>` arg yields path candidates.
        assert!(completion_candidates(None, "load", &format!("{p}/star"))
            .iter()
            .all(|c| c.contains("start")));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
