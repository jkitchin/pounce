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
use std::collections::{BTreeMap, HashSet};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::rc::Rc;

/// All command verbs, for `help` and `complete`.
const COMMANDS: &[&str] = &[
    "help", "info", "print", "step", "stepi", "continue", "run", "break", "stop-at", "set", "opt",
    "complete", "viz", "save", "goto", "restart", "resolve", "detach", "quit",
];

/// Checkpoint names a user can `stop-at` (matches `Checkpoint::as_str`).
const CHECKPOINTS: &[&str] = &[
    "iter_start",
    "after_mu",
    "after_search_dir",
    "after_step",
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
const METRICS: &[&str] = &["mu", "inf_pr", "inf_du", "obj", "err", "iter"];

/// A scalar the solver exposes for conditional breakpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    Mu,
    InfPr,
    InfDu,
    Obj,
    NlpError,
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
            // Tolerant equality so float metrics aren't impossible to hit.
            CmpOp::Eq => (lhs - rhs).abs() <= 1e-12 * rhs.abs().max(1.0),
        }
    }
}

/// A conditional breakpoint: pause when `metric op rhs` holds.
#[derive(Clone, Debug)]
struct Condition {
    metric: Metric,
    op: CmpOp,
    rhs: f64,
    /// Normalized source text, e.g. `inf_pr<1e-6`, for display / dedup.
    raw: String,
}

impl Condition {
    /// Parse a single `metric<op>value` expression (whitespace already
    /// stripped by the caller). Operators: `<`, `<=`, `>`, `>=`, `==`.
    fn parse(expr: &str) -> Result<Condition, String> {
        let expr = expr.trim();
        // Longest operators first so `<=` isn't truncated to `<`.
        let (op, pos, oplen) = ["<=", ">=", "==", "<", ">"]
            .iter()
            .find_map(|o| expr.find(o).map(|p| (*o, p, o.len())))
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
        Ok(Condition {
            metric,
            op: cmp,
            rhs,
            raw: format!("{metric_s}{op}{rhs_s}"),
        })
    }

    fn holds(&self, ctx: &DebugCtx) -> bool {
        self.op.eval(self.metric.eval(ctx), self.rhs)
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
        ["stop-at"] | ["stopat"] => starts(CHECKPOINTS),
        ["break", "if"] | ["b", "if"] => starts(METRICS),
        ["break"] | ["b"] => starts(&["if", "clear", "del"]),
        ["print"] | ["p"] => {
            let mut v = starts(&BLOCK_NAMES);
            v.extend(starts(&["mu", "obj", "inf_pr", "inf_du", "err", "iter"]));
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
    /// Conditional breakpoints (`break if mu<1e-4`): pause when any holds.
    conds: Vec<Condition>,
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
    /// One-shot: pause at the very next checkpoint of *any* kind (set by
    /// `stepi`, for walking through sub-iteration phases).
    sub_step: bool,
    /// Checkpoint kinds (by name) to always pause at (`stop-at`).
    stop_at: HashSet<&'static str>,
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
            conds: Vec::new(),
            detached: false,
            hello_sent: false,
            pause_iters: true,
            pause_terminal: true,
            terminal_only_on_error: false,
            interruptible: true,
            sub_step: false,
            stop_at: HashSet::new(),
            snapshots: BTreeMap::new(),
            restart: None,
            editor: None,
            hist_path: None,
            staged: Vec::new(),
        }
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
        self.breaks.contains(&iter)
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
            "stop-at" | "stopat" => self.cmd_stop_at(rest),
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
            "detach" => {
                self.detached = true;
                self.step = false;
                self.run_to = None;
                CmdOut::ok(vec!["detached — solving to completion".into()]).flow(Flow::Resume)
            }
            "quit" | "q" | "exit" => CmdOut::ok(vec!["stopping solve".into()]).flow(Flow::Stop),
            other => CmdOut::err(format!("unknown command `{other}` (try `help`)")),
        }
    }

    fn cmd_help(&self) -> CmdOut {
        let lines = vec![
            "commands:".into(),
            "  info | i                 summary of the current iterate".into(),
            "  print | p <what>         x|s|y_c|y_d|z_l|z_u|v_l|v_u | dx (step) |".into(),
            "                           mu|obj|inf_pr|inf_du|err|iter".into(),
            "  step | s | n             run one iteration, pause again".into(),
            "  stepi | si               run to the next checkpoint (into sub-iteration phases)".into(),
            "  stop-at <cp>             always pause at a checkpoint: after_mu|after_search_dir|after_step".into(),
            "  continue | c             run to the next breakpoint".into(),
            "  run | r <N>              run until iteration N".into(),
            "  break | b [N|clear|del N] set/list/clear breakpoints".into(),
            "  break if <m><op><v>      conditional bp; m in mu|inf_pr|inf_du|obj|err|iter,".into(),
            "                           op in < <= > >= ==  (e.g. break if inf_pr<1e-6)".into(),
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
                "iter" => ctx.iter() as f64,
                _ => {
                    return CmdOut::err(format!(
                        "don't know how to print `{what}` (try a block name or mu|obj|inf_pr|inf_du|err|iter)"
                    ))
                }
            };
            CmdOut::ok(vec![format!("{what} = {val:.10e}")])
                .with_data(serde_json::json!({"name": what, "value": val}))
        }
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
        match rest {
            [] => {
                let mut bs = self.breaks.clone();
                bs.sort_unstable();
                let conds: Vec<String> = self.conds.iter().map(|c| c.raw.clone()).collect();
                let mut lines = vec![format!("breakpoints: {bs:?}")];
                if !conds.is_empty() {
                    lines.push(format!("conditions: {}", conds.join(", ")));
                }
                CmdOut::ok(lines)
                    .with_data(serde_json::json!({"breakpoints": bs, "conditions": conds}))
            }
            ["clear", "cond"] | ["clear", "conditions"] => {
                self.conds.clear();
                CmdOut::ok(vec!["cleared conditional breakpoints".into()])
            }
            ["clear"] => {
                self.breaks.clear();
                self.conds.clear();
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

    fn cmd_complete(&self, rest: &[&str]) -> CmdOut {
        let prefix = rest.first().copied().unwrap_or("");
        let mut cands: Vec<String> = COMMANDS
            .iter()
            .filter(|c| c.starts_with(prefix))
            .map(|c| c.to_string())
            .collect();
        // Block names and option names also complete (handy after `set`).
        for b in BLOCK_NAMES {
            if b.starts_with(prefix) {
                cands.push(b.to_string());
            }
        }
        if let Some(reg) = self.reg.as_ref() {
            if prefix.len() >= 2 {
                for o in reg.registered_options_in_order() {
                    if o.name.starts_with(prefix) {
                        cands.push(o.name.clone());
                    }
                }
            }
        }
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

    fn cmd_viz(&self, rest: &[&str], ctx: &DebugCtx) -> CmdOut {
        let Some(&target) = rest.first() else {
            return CmdOut::err("usage: viz <x|s|y_c|...|dx|kkt|L>");
        };
        // Live KKT / L factor are assembled inside the search-direction
        // solver during compute_search_direction, not at the iteration
        // checkpoint, so they are not reachable from here yet.
        if target == "kkt" || target == "L" {
            return CmdOut::err(
                "live KKT/L visualization needs the factored augmented system, which isn't \
                 exposed at the iteration checkpoint yet. For now, re-run with \
                 `--dump kkt:<iter>+L+Lvals --dump-dir DIR` and open the per-iter dump.",
            );
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
                    eprintln!(
                        "\n── pounce-dbg ── iter {} @{}  mu={:.3e}  obj={:.6e}  inf_pr={:.2e}  inf_du={:.2e}",
                        ctx.iter(),
                        ctx.checkpoint().as_str(),
                        ctx.mu(),
                        ctx.objective(),
                        ctx.inf_pr(),
                        ctx.inf_du(),
                    );
                }
                if let Some(r) = reason {
                    eprintln!("   ↳ {r}");
                }
            }
            DebugMode::Json => {
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
                });
                emit_json(&ev);
            }
        }
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
                "conditional_breakpoints": true,
                "request_ids": true,
                "viz": ["block", "delta"],
                "save": true,
                "rewind": "primal_dual",
                "resolve": self.restart.is_some(),
                "terminal_checkpoint": true,
                "interruptible": self.interruptible,
            },
            "checkpoints": CHECKPOINTS,
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
        }
        read_stdin_line()
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
        }

        // Decide whether to pause. `stop-at` and a one-shot `stepi` apply
        // at every checkpoint; step / run / breakpoints / conditions /
        // Ctrl-C only at the iteration top.
        let mut reason: Option<String> = None;
        let mut pause = self.sub_step || self.stop_at.contains(cp.as_str());

        if is_iter_start {
            if self.interruptible && interrupt::take() {
                pause = true;
                reason = Some("interrupt (Ctrl-C)".into());
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
        }

        if !pause {
            return DebugAction::Resume;
        }
        // Consume one-shot arming; commands re-arm as needed.
        self.step = false;
        self.sub_step = false;
        self.ensure_editor();
        self.emit_pause(ctx, reason.as_deref());
        self.prompt_loop(ctx)
    }
}

impl SolverDebugger {
    /// Read and dispatch commands until one resumes or stops the solve.
    fn prompt_loop(&mut self, ctx: &mut DebugCtx) -> DebugAction {
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
    let dir = std::env::temp_dir();
    let path = dir.join(format!("pounce-dbg-{label}-iter{iter}.json"));
    let payload = serde_json::json!({"label": label, "iter": iter, "values": vals});
    std::fs::write(&path, payload.to_string()).map_err(|e| format!("write failed: {e}"))?;
    let path_s = path.to_string_lossy().to_string();

    let (program, args): (String, Vec<String>) = match std::env::var("POUNCE_DBG_VIEWER") {
        Ok(tmpl) if !tmpl.trim().is_empty() => {
            let mut parts = tmpl
                .split_whitespace()
                .map(|s| s.to_string())
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
            (prog, parts)
        }
        _ => {
            let prog = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            (prog.to_string(), vec![path_s.clone()])
        }
    };
    let viewer = format!("{program} {}", args.join(" "));
    match std::process::Command::new(&program).args(&args).spawn() {
        Ok(_) => Ok((path_s, viewer)),
        Err(e) => Err(format!(
            "wrote {path_s} but could not launch `{program}`: {e} \
             (set POUNCE_DBG_VIEWER to a command, e.g. `python my_plot.py {{}}`)"
        )),
    }
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
    fn condition_parses_metric_op_threshold() {
        let c = Condition::parse("mu<1e-4").unwrap();
        assert_eq!(c.metric, Metric::Mu);
        assert_eq!(c.op, CmpOp::Lt);
        assert_eq!(c.rhs, 1e-4);

        // `<=` must not be truncated to `<`, and spaces are tolerated by
        // the caller (tokens are concatenated before parse).
        let c = Condition::parse("inf_pr<=1e-6").unwrap();
        assert_eq!(c.metric, Metric::InfPr);
        assert_eq!(c.op, CmpOp::Le);

        let c = Condition::parse("iter==10").unwrap();
        assert_eq!(c.metric, Metric::Iter);
        assert_eq!(c.op, CmpOp::Eq);
        assert_eq!(c.rhs, 10.0);
    }

    #[test]
    fn condition_parse_rejects_garbage() {
        assert!(Condition::parse("inf_pr 1e-6").is_err()); // no operator
        assert!(Condition::parse("bogus<1").is_err()); // unknown metric
        assert!(Condition::parse("mu<abc").is_err()); // bad threshold
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
    fn detach_disables_all_pausing() {
        let mut d = dbg(DebugMode::Repl);
        d.detached = true;
        d.step = true;
        d.breaks = vec![1];
        assert!(!d.should_pause(0));
        assert!(!d.should_pause(1));
    }
}
