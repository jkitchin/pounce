//! Diagnostic-dump infrastructure shared by the solver and the CLI.
//!
//! # Why this exists
//!
//! Debugging a stalled solve or a perf regression usually means
//! capturing the inner state of the IPM at specific iterations:
//! the augmented-system KKT matrix, the iterate, the search step,
//! the line-search trace. Historically this lived as a scatter of
//! `POUNCE_DBG_*` env-vars across the codebase, each with bespoke
//! semantics. This module centralizes the surface so the CLI
//! (`--dump kkt:5-10`) and the dump sites (deep in the linear
//! solver) speak the same vocabulary.
//!
//! # Lifecycle
//!
//! 1. The CLI parses `--dump <cat>[:<spec>]` flags into a
//!    [`DiagnosticsConfig`] and constructs a [`DiagnosticsState`].
//! 2. The application installs the state via
//!    `IpoptApplication::set_diagnostics`, then propagates an
//!    `Rc<DiagnosticsState>` through the solve in the same way
//!    [`crate::timing::TimingStatistics`] is propagated.
//! 3. At the top of each outer iteration, the IPM calls
//!    [`DiagnosticsState::bump_iter`] to advance the current-iter
//!    counter and reset the per-iter solve index.
//! 4. Every dump site (KKT solver, line search, μ oracle, ...) calls
//!    [`DiagnosticsState::want`] to gate the dump, then
//!    [`DiagnosticsState::open_writer`] to obtain a file handle in
//!    the right `iter_NNN/` sub-directory.
//!
//! # File layout
//!
//! ```text
//! <dump_dir>/
//!   manifest.json
//!   iter_005/
//!     kkt_solve_001.jsonl
//!     iterate.json
//!   iter_006/...
//!   resto/
//!     parent_iter_007/
//!       iter_000/kkt_solve_001.jsonl
//!   timing.json
//! ```
//!
//! The `solve_NNN` suffix disambiguates the multi-solve-per-iter case
//! (second-order corrections and perturbation re-solves issue extra
//! factorizations inside one outer iteration). The
//! `resto/parent_iter_NNN/` hierarchy keeps the restoration sub-IPM
//! trace separate from the main solve trace.

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

/// Single diagnostic category the user can request.
///
/// Categories map roughly one-to-one to dump sites in the solver.
/// `Kkt` is the only one actually wired in PR-A; the rest are
/// declared up front so `--dump iterate:all` parses today and the
/// follow-up PRs only need to flip the dump-site switch.
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum DiagCategory {
    Kkt,
    Iterate,
    Step,
    Mu,
    Ls,
    Resto,
    Convergence,
    Timing,
}

impl DiagCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            DiagCategory::Kkt => "kkt",
            DiagCategory::Iterate => "iterate",
            DiagCategory::Step => "step",
            DiagCategory::Mu => "mu",
            DiagCategory::Ls => "ls",
            DiagCategory::Resto => "resto",
            DiagCategory::Convergence => "convergence",
            DiagCategory::Timing => "timing",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "kkt" => Ok(DiagCategory::Kkt),
            "iterate" => Ok(DiagCategory::Iterate),
            "step" => Ok(DiagCategory::Step),
            "mu" => Ok(DiagCategory::Mu),
            "ls" => Ok(DiagCategory::Ls),
            "resto" => Ok(DiagCategory::Resto),
            "convergence" => Ok(DiagCategory::Convergence),
            "timing" => Ok(DiagCategory::Timing),
            other => Err(format!(
                "unknown dump category '{other}' (expected one of: kkt, iterate, step, mu, ls, resto, convergence, timing)"
            )),
        }
    }
}

/// Iteration filter attached to a category. `None` endpoints denote
/// open-ended ranges (`N-` is `Range(Some(N), None)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterSpec {
    All,
    Single(i32),
    Range(Option<i32>, Option<i32>),
}

impl IterSpec {
    pub fn includes(&self, iter: i32) -> bool {
        match self {
            IterSpec::All => true,
            IterSpec::Single(n) => iter == *n,
            IterSpec::Range(lo, hi) => {
                lo.map_or(true, |l| iter >= l) && hi.map_or(true, |h| iter <= h)
            }
        }
    }

    /// Parse the grammar `all | N | N-M | N- | -M`. Negative ints
    /// aren't accepted — iter counts are non-negative by definition.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() || s == "all" {
            return Ok(IterSpec::All);
        }
        if let Some(rest) = s.strip_prefix('-') {
            // "-M"
            let hi: i32 = rest
                .parse()
                .map_err(|_| format!("invalid iter-spec '{s}': expected '-M' with non-negative integer M"))?;
            if hi < 0 {
                return Err(format!("invalid iter-spec '{s}': iter must be non-negative"));
            }
            return Ok(IterSpec::Range(None, Some(hi)));
        }
        if let Some((a, b)) = s.split_once('-') {
            let lo: i32 = a
                .parse()
                .map_err(|_| format!("invalid iter-spec '{s}': '{a}' is not an integer"))?;
            if lo < 0 {
                return Err(format!("invalid iter-spec '{s}': iter must be non-negative"));
            }
            if b.is_empty() {
                // "N-"
                return Ok(IterSpec::Range(Some(lo), None));
            }
            // "N-M"
            let hi: i32 = b
                .parse()
                .map_err(|_| format!("invalid iter-spec '{s}': '{b}' is not an integer"))?;
            if hi < 0 {
                return Err(format!("invalid iter-spec '{s}': iter must be non-negative"));
            }
            if hi < lo {
                return Err(format!(
                    "invalid iter-spec '{s}': end ({hi}) is below start ({lo})"
                ));
            }
            return Ok(IterSpec::Range(Some(lo), Some(hi)));
        }
        // Bare "N"
        let n: i32 = s
            .parse()
            .map_err(|_| format!("invalid iter-spec '{s}': expected 'all', 'N', 'N-M', 'N-', or '-M'"))?;
        if n < 0 {
            return Err(format!("invalid iter-spec '{s}': iter must be non-negative"));
        }
        Ok(IterSpec::Single(n))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DumpFormat {
    /// Newline-delimited JSON records. One record per dump call.
    /// Hackable from a shell one-liner; the default.
    Jsonl,
}

impl DumpFormat {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "jsonl" => Ok(DumpFormat::Jsonl),
            other => Err(format!(
                "unknown dump format '{other}' (expected: jsonl)"
            )),
        }
    }
}

/// Static configuration: where to dump, in what format, with what
/// per-category iter filters. Constructed by the CLI, held by the
/// application, frozen for the duration of a solve.
#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    pub dump_dir: PathBuf,
    pub format: DumpFormat,
    pub categories: HashMap<DiagCategory, IterSpec>,
}

impl DiagnosticsConfig {
    pub fn new(dump_dir: PathBuf) -> Self {
        Self {
            dump_dir,
            format: DumpFormat::Jsonl,
            categories: HashMap::new(),
        }
    }

    pub fn with_category(mut self, cat: DiagCategory, spec: IterSpec) -> Self {
        self.categories.insert(cat, spec);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }
}

/// Live state threaded through the solve via `Rc`. The IPM mutates
/// `current_iter` and `solves_this_iter`; the dump sites read them.
/// All fields use atomics so the type is `Send + Sync` even though
/// the solver itself is single-threaded — keeps the door open for
/// future parallel inner solvers without an ABI rewrite.
pub struct DiagnosticsState {
    pub config: DiagnosticsConfig,
    current_iter: AtomicI32,
    solves_this_iter: AtomicI32,
    in_restoration: AtomicBool,
    resto_parent_iter: AtomicI32,
    resto_inner_iter: AtomicI32,
    resto_solves_this_iter: AtomicI32,
}

impl DiagnosticsState {
    /// Create a state and `mkdir -p` the dump directory. Failure to
    /// create the directory bubbles up so the CLI can exit with a
    /// clear error before the solve even starts.
    pub fn new(config: DiagnosticsConfig) -> std::io::Result<Self> {
        fs::create_dir_all(&config.dump_dir)?;
        Ok(Self {
            config,
            current_iter: AtomicI32::new(-1),
            solves_this_iter: AtomicI32::new(0),
            in_restoration: AtomicBool::new(false),
            resto_parent_iter: AtomicI32::new(-1),
            resto_inner_iter: AtomicI32::new(-1),
            resto_solves_this_iter: AtomicI32::new(0),
        })
    }

    /// True if the caller should dump `cat` at the current iter.
    pub fn want(&self, cat: DiagCategory) -> bool {
        let iter = self.effective_iter();
        if iter < 0 {
            return false;
        }
        self.config
            .categories
            .get(&cat)
            .map(|spec| spec.includes(iter))
            .unwrap_or(false)
    }

    /// Advance the outer-iteration counter and reset the per-iter
    /// solve index. Called by `IpoptAlgorithm::optimize` at the top
    /// of each outer iteration.
    pub fn bump_iter(&self) {
        if self.in_restoration.load(Ordering::SeqCst) {
            self.resto_inner_iter.fetch_add(1, Ordering::SeqCst);
            self.resto_solves_this_iter.store(0, Ordering::SeqCst);
        } else {
            self.current_iter.fetch_add(1, Ordering::SeqCst);
            self.solves_this_iter.store(0, Ordering::SeqCst);
        }
    }

    /// Reserve the next per-iter solve index. Returned value is
    /// 1-based to match the filenames (`kkt_solve_001.jsonl`).
    pub fn next_solve_index(&self) -> i32 {
        let counter = if self.in_restoration.load(Ordering::SeqCst) {
            &self.resto_solves_this_iter
        } else {
            &self.solves_this_iter
        };
        counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Mark the start of a restoration sub-IPM run. `parent_iter` is
    /// the outer iter that triggered restoration; dumps from the
    /// resto sub-solve land under `resto/parent_iter_NNN/iter_MMM/`.
    pub fn enter_restoration(&self) {
        let parent = self.current_iter.load(Ordering::SeqCst);
        self.resto_parent_iter.store(parent, Ordering::SeqCst);
        self.resto_inner_iter.store(-1, Ordering::SeqCst);
        self.resto_solves_this_iter.store(0, Ordering::SeqCst);
        self.in_restoration.store(true, Ordering::SeqCst);
    }

    pub fn exit_restoration(&self) {
        self.in_restoration.store(false, Ordering::SeqCst);
    }

    pub fn current_iter(&self) -> i32 {
        self.effective_iter()
    }

    /// The iter counter that gates current dumps — resto inner iter
    /// when in restoration, main outer iter otherwise.
    fn effective_iter(&self) -> i32 {
        if self.in_restoration.load(Ordering::SeqCst) {
            self.resto_inner_iter.load(Ordering::SeqCst)
        } else {
            self.current_iter.load(Ordering::SeqCst)
        }
    }

    /// Resolve the directory a category's dump file should live in,
    /// creating it if necessary. Returns `None` if the directory
    /// cannot be created (e.g., filesystem full) — callers should
    /// silently skip the dump in that case rather than fail the
    /// solve.
    pub fn iter_dir(&self) -> Option<PathBuf> {
        let dir = if self.in_restoration.load(Ordering::SeqCst) {
            let parent = self.resto_parent_iter.load(Ordering::SeqCst);
            let inner = self.resto_inner_iter.load(Ordering::SeqCst).max(0);
            self.config.dump_dir.join(format!(
                "resto/parent_iter_{parent:03}/iter_{inner:03}"
            ))
        } else {
            let iter = self.current_iter.load(Ordering::SeqCst).max(0);
            self.config.dump_dir.join(format!("iter_{iter:03}"))
        };
        fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }

    /// Open a writer for `<iter_dir>/<filename>`. Caller picks the
    /// filename so callers that produce multi-solve traces can use
    /// `next_solve_index` to disambiguate.
    pub fn open_writer(&self, filename: &str) -> Option<BufWriter<fs::File>> {
        let dir = self.iter_dir()?;
        let path = dir.join(filename);
        fs::File::create(path).ok().map(BufWriter::new)
    }

    /// Write a one-shot top-level file (manifest, timing summary).
    /// Always lands directly under `dump_dir`, never under an iter
    /// sub-directory.
    pub fn write_top_level(&self, filename: &str, contents: &str) -> std::io::Result<()> {
        let path = self.config.dump_dir.join(filename);
        let mut f = fs::File::create(path)?;
        f.write_all(contents.as_bytes())?;
        f.flush()
    }

    pub fn dump_dir(&self) -> &Path {
        &self.config.dump_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_spec_parses_all_grammar_forms() {
        assert_eq!(IterSpec::parse("").unwrap(), IterSpec::All);
        assert_eq!(IterSpec::parse("all").unwrap(), IterSpec::All);
        assert_eq!(IterSpec::parse("5").unwrap(), IterSpec::Single(5));
        assert_eq!(
            IterSpec::parse("5-10").unwrap(),
            IterSpec::Range(Some(5), Some(10))
        );
        assert_eq!(
            IterSpec::parse("5-").unwrap(),
            IterSpec::Range(Some(5), None)
        );
        assert_eq!(
            IterSpec::parse("-10").unwrap(),
            IterSpec::Range(None, Some(10))
        );
    }

    #[test]
    fn iter_spec_rejects_malformed_input() {
        assert!(IterSpec::parse("abc").is_err());
        assert!(IterSpec::parse("5-3").is_err()); // end below start
        assert!(IterSpec::parse("-x").is_err());
        assert!(IterSpec::parse("5--10").is_err()); // doubled separator → "-10" tail parse fails
    }

    #[test]
    fn iter_spec_includes_matches_grammar() {
        assert!(IterSpec::All.includes(0));
        assert!(IterSpec::All.includes(1000));
        assert!(IterSpec::Single(5).includes(5));
        assert!(!IterSpec::Single(5).includes(4));
        let r = IterSpec::Range(Some(5), Some(10));
        assert!(!r.includes(4));
        assert!(r.includes(5));
        assert!(r.includes(7));
        assert!(r.includes(10));
        assert!(!r.includes(11));
        assert!(IterSpec::Range(Some(5), None).includes(1_000_000));
        assert!(IterSpec::Range(None, Some(5)).includes(0));
    }

    #[test]
    fn category_parses_known_names() {
        assert_eq!(DiagCategory::parse("kkt").unwrap(), DiagCategory::Kkt);
        assert_eq!(
            DiagCategory::parse("iterate").unwrap(),
            DiagCategory::Iterate
        );
        assert!(DiagCategory::parse("bogus").is_err());
    }

    #[test]
    fn state_gates_on_iter_spec() {
        let tmp = tempdir();
        let cfg = DiagnosticsConfig::new(tmp.clone())
            .with_category(DiagCategory::Kkt, IterSpec::Range(Some(2), Some(4)));
        let state = DiagnosticsState::new(cfg).unwrap();

        // Before bump_iter, current_iter == -1 → no dumps.
        assert!(!state.want(DiagCategory::Kkt));

        state.bump_iter(); // iter 0
        assert!(!state.want(DiagCategory::Kkt));
        state.bump_iter(); // 1
        assert!(!state.want(DiagCategory::Kkt));
        state.bump_iter(); // 2
        assert!(state.want(DiagCategory::Kkt));
        state.bump_iter(); // 3
        assert!(state.want(DiagCategory::Kkt));
        state.bump_iter(); // 4
        assert!(state.want(DiagCategory::Kkt));
        state.bump_iter(); // 5
        assert!(!state.want(DiagCategory::Kkt));

        // Other categories silently skipped (not configured).
        assert!(!state.want(DiagCategory::Iterate));

        fs::remove_dir_all(tmp).ok();
    }

    #[test]
    fn state_emits_solve_indices_and_iter_dirs() {
        let tmp = tempdir();
        let cfg = DiagnosticsConfig::new(tmp.clone())
            .with_category(DiagCategory::Kkt, IterSpec::All);
        let state = DiagnosticsState::new(cfg).unwrap();
        state.bump_iter(); // iter 0
        assert_eq!(state.next_solve_index(), 1);
        assert_eq!(state.next_solve_index(), 2);
        state.bump_iter(); // iter 1
        assert_eq!(state.next_solve_index(), 1);

        let dir = state.iter_dir().unwrap();
        assert!(dir.ends_with("iter_001"));
        fs::remove_dir_all(tmp).ok();
    }

    #[test]
    fn restoration_dumps_live_under_resto_subtree() {
        let tmp = tempdir();
        let cfg = DiagnosticsConfig::new(tmp.clone())
            .with_category(DiagCategory::Kkt, IterSpec::All);
        let state = DiagnosticsState::new(cfg).unwrap();
        state.bump_iter(); // main iter 0
        state.bump_iter(); // main iter 1
        state.enter_restoration();
        state.bump_iter(); // resto inner 0
        let dir = state.iter_dir().unwrap();
        assert!(
            dir.ends_with("resto/parent_iter_001/iter_000"),
            "got {dir:?}"
        );
        assert_eq!(state.next_solve_index(), 1);
        state.exit_restoration();
        let dir = state.iter_dir().unwrap();
        assert!(dir.ends_with("iter_001"), "got {dir:?}");
        fs::remove_dir_all(tmp).ok();
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "pounce-diag-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }
}
