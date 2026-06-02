//! Interactive solver debugger front end — "pdb for the IPM".
//!
//! Implements [`pounce_algorithm::debug::DebugHook`]. The core fires us
//! at every checkpoint (today: the top of each outer iteration); we
//! pause, hand the user (or an agent) a command prompt, and apply
//! inspect / mutate / flow commands against the live [`DebugCtx`] before
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
    Checkpoint, DebugAction, DebugCtx, DebugHook, IterateSnapshot, BLOCK_NAMES,
};
use pounce_common::reg_options::{DefaultValue, OptionType, RegisteredOptions};
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
    "opt",
    "complete",
    "viz",
    "save",
    "goto",
    "restart",
    "resolve",
    "ask",
    "watch",
    "diff",
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
    pub seed_x: Vec<f64>,
    /// `set opt` edits staged during the session, to apply before re-solve.
    pub options: Vec<(String, String)>,
}

/// Shared slot the debugger uses to hand a [`RestartRequest`] back to the
/// CLI's re-solve loop.
pub type RestartCell = Rc<std::cell::RefCell<Option<RestartRequest>>>;

/// Cap on retained per-iteration snapshots (bounds rewind memory; oldest
/// are evicted first).
const SNAPSHOT_CAP: usize = 2000;

/// SolverReturn debug strings that count as a successful solve (so
/// `--debug-on-error` does *not* pause at the terminal checkpoint).
fn is_success_status(s: &str) -> bool {
    matches!(s, "Success" | "StopAtAcceptablePoint")
}

/// SIGINT → "break into the debugger at the next iteration". A first
/// Ctrl-C sets a pending flag the hook consumes at the next checkpoint;
/// a second Ctrl-C before that (or any Ctrl-C once detached) hard-exits,
/// preserving the usual "abort" escape hatch.
///
/// At a rustyline prompt the terminal is in raw mode, so Ctrl-C arrives
/// as input (handled as `Interrupted`, reprompt) rather than as SIGINT —
/// this handler only fires while the solve is running.
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
    fn eval(self, ctx: &DebugCtx) -> f64 {
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

    fn holds(&self, ctx: &DebugCtx) -> bool {
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

    fn holds(&self, ctx: &DebugCtx) -> bool {
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
        ["set", "opt"] | ["opt"] | ["options"] => opt_names(),
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
                "mu", "obj", "inf_pr", "inf_du", "err", "compl", "iter", "kkt", "active",
            ]));
            v
        }
        ["viz"] | ["plot"] => {
            let mut v = starts(&BLOCK_NAMES);
            v.extend(starts(&["kkt", "L"]));
            v
        }
        ["complete"] => starts(COMMANDS),
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
    snapshots: BTreeMap<i32, IterateSnapshot>,
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
        }
    }

    /// Queue a debugger script to run once at the first pause.
    pub fn with_script(mut self, path: String) -> Self {
        self.pending_script = Some(path);
        self
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
    fn matched_condition(&self, ctx: &DebugCtx) -> Option<String> {
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
    fn matched_event(&self, ctx: &DebugCtx) -> Option<&'static str> {
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
    fn matched_watchpoint(&mut self, ctx: &DebugCtx) -> Option<String> {
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

    fn dispatch(&mut self, line: &str, ctx: &mut DebugCtx) -> CmdOut {
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
            "watchpoint" | "wp" => self.cmd_watchpoint(rest),
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
            "opt" | "options" => self.cmd_opt(rest),
            "complete" => self.cmd_complete(rest),
            "viz" | "plot" => self.cmd_viz(rest, ctx),
            "save" => self.cmd_save(rest, ctx),
            "goto" | "jump" => self.cmd_goto(rest, ctx),
            "restart" => match self.snapshots.keys().next().copied() {
                Some(k) => self.restore_to(k, ctx),
                None => CmdOut::err("no snapshots captured yet"),
            },
            "resolve" | "re-solve" => self.cmd_resolve(ctx),
            "ask" | "explain" | "claude" => self.cmd_ask(rest, ctx),
            "watch" | "display" => self.cmd_watch(rest),
            "diff" => self.cmd_diff(ctx),
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
            "quit" | "q" | "exit" => CmdOut::ok(vec!["stopping solve".into()]).flow(Flow::Stop),
            other => CmdOut::err(format!("unknown command `{other}` (try `help`)")),
        }
    }

    fn cmd_help(&self) -> CmdOut {
        let lines = vec![
            "commands:".into(),
            "  info | i                 summary of the current iterate".into(),
            "  print | p <what>         x|s|y_c|y_d|z_l|z_u|v_l|v_u | dx (step) |".into(),
            "                           mu|obj|inf_pr|inf_du|err|compl|iter | kkt | active".into(),
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
            "  opt [filter]             list solver options (name/type/default)".into(),
            "  complete <prefix>        completion candidates (commands + options)".into(),
            "  viz <x|s|dx|...|kkt|L>   open the artifact in an external viewer".into(),
            "  save [path]              write the current iterate + residuals to JSON".into(),
            "  goto <k> | restart       rewind to a captured iteration (primal-dual only)".into(),
            "  resolve                  re-solve from the current x with staged `set opt`s".into(),
            "  ask [question]           ask Claude Code (claude -p / $POUNCE_DBG_LLM) about the state".into(),
            "  watch [target|clear|del] auto-print a `print` target at every pause".into(),
            "  diff                     what changed in the iterate since the last iteration".into(),
            "  source <file>            run debugger commands from a file".into(),
            "  detach                   stop pausing; solve to completion".into(),
            "  quit | q                 stop the solve now".into(),
        ];
        CmdOut::ok(lines)
    }

    fn cmd_info(&self, ctx: &DebugCtx) -> CmdOut {
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

    fn cmd_print(&self, rest: &[&str], ctx: &DebugCtx) -> CmdOut {
        let Some(&what) = rest.first() else {
            return self.cmd_info(ctx);
        };
        if what == "kkt" {
            return self.cmd_print_kkt(ctx);
        }
        if what == "active" {
            return self.cmd_print_active(ctx);
        }
        // step / delta blocks: `dx`, `ds`, ... or `delta_x`.
        let delta = what.strip_prefix("d").filter(|b| BLOCK_NAMES.contains(b));
        if BLOCK_NAMES.contains(&what) {
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

    /// `print active` — bound-slack classification: how many of each
    /// bound category are near-active (slack below `tol`), and the min
    /// slack. Small slacks mark the bounds the iterate is pressing on.
    fn cmd_print_active(&self, ctx: &DebugCtx) -> CmdOut {
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
            let min = sl.iter().copied().fold(f64::INFINITY, f64::min);
            let near = sl.iter().filter(|&&s| s.abs() < tol).count();
            lines.push(format!(
                "{cat}: {n} bound(s), {near} near-active (slack<{tol:.0e}), min slack {min:.3e}"
            ));
            cats.insert(
                cat.to_string(),
                serde_json::json!({"n": n, "near_active": near, "min_slack": min}),
            );
        }
        if lines.is_empty() {
            lines.push("no bounded variables or inequality slacks".into());
        }
        CmdOut::ok(lines).with_data(serde_json::json!({"tol": tol, "categories": cats}))
    }

    /// `print kkt` — inertia + regularization of the factored augmented
    /// system. Only meaningful at/after `after_search_dir`.
    fn cmd_print_kkt(&self, ctx: &DebugCtx) -> CmdOut {
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

    fn cmd_set(&mut self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
        match rest {
            ["mu", v] => match v.parse::<f64>() {
                Ok(mu) => match ctx.set_mu(mu) {
                    Ok(()) => CmdOut::ok(vec![format!("mu := {mu:.6e}")]),
                    Err(e) => CmdOut::err(e),
                },
                Err(_) => CmdOut::err("usage: set mu <value>"),
            },
            ["opt", name, value] => self.cmd_set_opt(name, value),
            [target, value] => self.cmd_set_block(target, value, ctx),
            _ => CmdOut::err(
                "usage: set mu <v> | set <blk>[<i>] <v> | set <blk> <v0,v1,..> | set opt <name> <v>",
            ),
        }
    }

    /// `set x[2] 1.5` (component) or `set x 1,2,3` (whole block).
    fn cmd_set_block(&mut self, target: &str, value: &str, ctx: &mut DebugCtx) -> CmdOut {
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

    fn cmd_set_opt(&mut self, name: &str, value: &str) -> CmdOut {
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
        self.staged.retain(|(k, _)| k != name);
        self.staged.push((name.to_string(), value.to_string()));
        CmdOut::ok(vec![format!(
            "staged {name} = {value}  (validated; applied on next solve — built strategies don't re-read mid-solve)"
        )])
        .with_data(serde_json::json!({"option": name, "value": value, "staged": true}))
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
    fn cmd_save(&self, rest: &[&str], ctx: &DebugCtx) -> CmdOut {
        let iter = ctx.iter();
        let path = rest
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join(format!("pounce-dbg-iter{iter}.json")));
        let collect = |delta: bool| -> serde_json::Map<String, serde_json::Value> {
            let mut m = serde_json::Map::new();
            for &b in BLOCK_NAMES.iter() {
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

    /// `goto <k>` — rewind to a captured iteration.
    fn cmd_goto(&mut self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
        match rest.first().and_then(|s| s.parse::<i32>().ok()) {
            Some(k) => self.restore_to(k, ctx),
            None => CmdOut::err("usage: goto <iteration>"),
        }
    }

    /// Restore the snapshot for iteration `k` (primal-dual state only;
    /// strategy history is not rewound). Stays paused so the user can
    /// inspect / re-tune before `continue`/`step`.
    fn restore_to(&mut self, k: i32, ctx: &mut DebugCtx) -> CmdOut {
        match self.snapshots.get(&k) {
            Some(snap) => {
                ctx.restore(snap);
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

    /// `resolve` — capture the current primal `x` and the staged option
    /// edits, then stop this solve so the CLI re-runs from that point with
    /// the new options applied (a primal warm start). Needs a restart cell
    /// (wired by the CLI); a no-op error otherwise.
    fn cmd_resolve(&mut self, ctx: &DebugCtx) -> CmdOut {
        let Some(cell) = self.restart.as_ref() else {
            return CmdOut::err("re-solve is not available in this context");
        };
        let Some(seed_x) = ctx.block("x") else {
            return CmdOut::err("no current iterate to seed from");
        };
        let options = self.staged.clone();
        let n_opt = options.len();
        *cell.borrow_mut() = Some(RestartRequest { seed_x, options });
        CmdOut::ok(vec![format!(
            "re-solving from current x with {n_opt} staged option override(s)…"
        )])
        .with_data(serde_json::json!({"resolve": true, "options": n_opt}))
        .flow(Flow::Stop)
    }

    /// `ask [question]` — hand the current solver state to Claude Code
    /// (headless `claude -p`, or `$POUNCE_DBG_LLM`) and print its reply.
    /// "Ask why this step looks wrong without leaving the debugger."
    fn cmd_ask(&self, rest: &[&str], ctx: &DebugCtx) -> CmdOut {
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
    fn cmd_watchpoint(&mut self, rest: &[&str]) -> CmdOut {
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
                if !BLOCK_NAMES.contains(&block.as_str()) {
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
    fn cmd_diff(&self, ctx: &DebugCtx) -> CmdOut {
        let iter = ctx.iter();
        let Some((&piter, prev)) = self.snapshots.range(..iter).next_back() else {
            return CmdOut::err("no previous iterate to diff against");
        };
        let mut lines = vec![format!("Δ since iter {piter}:")];
        let dmu = ctx.mu() - prev.mu();
        lines.push(format!("  mu  = {:.6e}  (Δ {:+.3e})", ctx.mu(), dmu));
        let mut blocks = serde_json::Map::new();
        for b in BLOCK_NAMES {
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
    fn cmd_source(&mut self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
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

    fn cmd_viz(&self, rest: &[&str], ctx: &mut DebugCtx) -> CmdOut {
        let Some(&target) = rest.first() else {
            return CmdOut::err("usage: viz <x|s|y_c|...|dx|kkt|L>");
        };
        // `viz kkt` writes the assembled augmented-system matrix (triplets
        // → heatmap) plus the inertia/regularization summary.
        if target == "kkt" {
            let Some(k) = ctx.kkt() else {
                return CmdOut::err(
                    "no KKT factorization yet — stop at `after_search_dir` (e.g. `stop-at kkt`)",
                );
            };
            // Triplet capture is opt-in (O(nnz) assembly), so the first call
            // arms it — same dance as `viz L`.
            let Some((dim, irn, jcn, vals)) = ctx.kkt_matrix() else {
                ctx.request_kkt_matrix();
                return CmdOut::err(
                    "KKT matrix capture enabled — re-run `viz kkt` after the next \
                     `after_search_dir` stop (`stepi` or `continue`).",
                );
            };
            let matrix = serde_json::json!({"dim": dim, "irn": irn, "jcn": jcn, "vals": vals,
                                            "format": "triplet_1based_lower"});
            let payload = serde_json::json!({
                "label": "kkt", "iter": ctx.iter(),
                "dim": k.dim, "n_pos": k.n_pos, "n_neg": k.n_neg,
                "expected_neg": k.expected_neg, "inertia_correct": k.inertia_correct,
                "delta_w": k.delta_w, "delta_c": k.delta_c, "status": k.status,
                "matrix": matrix,
            });
            return match write_json_and_open("kkt", ctx.iter(), &payload) {
                Ok((path, viewer)) => {
                    CmdOut::ok(vec![format!("wrote {path}; opened with `{viewer}`")])
                        .with_data(serde_json::json!({"path": path, "viewer": viewer}))
                }
                Err(e) => CmdOut::err(e),
            };
        }
        // `viz L` writes the LDLᵀ factor triplets. Capture is opt-in (the
        // factor is the expensive piece), so the first call arms it.
        if target == "L" {
            match ctx.kkt_l_factor() {
                Some((n, perm, l_irn, l_jcn, l_vals)) => {
                    let payload = serde_json::json!({
                        "label": "L", "iter": ctx.iter(), "n": n, "perm": perm,
                        "l_irn": l_irn, "l_jcn": l_jcn, "l_vals": l_vals,
                        "format": "strict_lower_1based_permuted",
                    });
                    return match write_json_and_open("L", ctx.iter(), &payload) {
                        Ok((path, viewer)) => {
                            CmdOut::ok(vec![format!("wrote {path}; opened with `{viewer}`")])
                                .with_data(serde_json::json!({"path": path, "viewer": viewer}))
                        }
                        Err(e) => CmdOut::err(e),
                    };
                }
                None => {
                    ctx.request_l_factor();
                    return CmdOut::err(
                        "L-factor capture enabled — re-run `viz L` after the next \
                         `after_search_dir` stop (`stepi` or `continue`).",
                    );
                }
            }
        }
        // Resolve the vector to visualize.
        let (label, vals) = if BLOCK_NAMES.contains(&target) {
            match ctx.block(target) {
                Some(v) => (target.to_string(), v),
                None => return CmdOut::err(format!("no data for block `{target}`")),
            }
        } else if let Some(blk) = target.strip_prefix("d").filter(|b| BLOCK_NAMES.contains(b)) {
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
    fn emit_pause(&self, ctx: &DebugCtx, reason: Option<&str>) {
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
    fn emit_progress_event(&self, ctx: &DebugCtx) {
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
                "kkt_inspect": true,
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

    /// Read one command line. Returns `None` on EOF. Uses rustyline when
    /// an editor is active (history / Tab / Ctrl-R); otherwise a plain
    /// reader with a stderr prompt (REPL) or no prompt (JSON).
    fn next_command_line(&mut self) -> Option<String> {
        if let DebugMode::Repl = self.mode {
            if let Some(ed) = self.editor.as_mut() {
                return match ed.readline("pounce-dbg> ") {
                    Ok(l) => {
                        let _ = ed.add_history_entry(l.as_str());
                        if let Some(p) = &self.hist_path {
                            let _ = ed.save_history(p);
                        }
                        Some(l)
                    }
                    // Ctrl-C: abandon the current line, reprompt.
                    Err(ReadlineError::Interrupted) => Some(String::new()),
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
    fn at_checkpoint(&mut self, ctx: &mut DebugCtx) -> DebugAction {
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
                self.snapshots.insert(snap.iter(), snap);
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
    fn prompt_loop(&mut self, ctx: &mut DebugCtx) -> DebugAction {
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
fn build_ask_prompt(ctx: &DebugCtx, question: &str) -> String {
    use std::fmt::Write as _;
    let mut p = String::new();
    p.push_str(
        "You are helping debug a paused run of POUNCE, a pure-Rust port of the Ipopt \
         interior-point NLP solver. The solve is stopped at a debugger checkpoint. \
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

    // Candidate viewers, tried in order until one launches:
    //   1. $POUNCE_DBG_VIEWER (a command template; `{}` ← the path),
    //   2. `pounce-dbg-viz` — the bundled interactive Plotly viewer
    //      (`pip install 'pounce-solver[viz]'`), when on PATH,
    //   3. the OS opener (xdg-open / open) on the raw JSON.
    let mut candidates: Vec<(String, Vec<String>)> = Vec::new();
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
            candidates.push((prog, parts));
        }
        _ => {
            candidates.push(("pounce-dbg-viz".to_string(), vec![path_s.clone()]));
            let opener = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            candidates.push((opener.to_string(), vec![path_s.clone()]));
        }
    }

    let mut last_err = String::new();
    for (program, args) in &candidates {
        match std::process::Command::new(program).args(args).spawn() {
            Ok(_) => return Ok((path_s, format!("{program} {}", args.join(" ")))),
            Err(e) => last_err = format!("`{program}`: {e}"),
        }
    }
    Err(format!(
        "wrote {path_s} but could not launch a viewer ({last_err}). \
         Install the interactive viewer (`pip install 'pounce-solver[viz]'`) \
         or set POUNCE_DBG_VIEWER, e.g. `python my_plot.py {{}}`."
    ))
}

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
}
