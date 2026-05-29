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

use std::cell::RefCell;
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
            // Accept both "iterate" (legacy) and "iterates" (the
            // public name from issue #68's contract). Internally one
            // enum variant.
            "iterate" | "iterates" => Ok(DiagCategory::Iterate),
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
            IterSpec::Range(lo, hi) => lo.is_none_or(|l| iter >= l) && hi.is_none_or(|h| iter <= h),
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
            let hi: i32 = rest.parse().map_err(|_| {
                format!("invalid iter-spec '{s}': expected '-M' with non-negative integer M")
            })?;
            if hi < 0 {
                return Err(format!(
                    "invalid iter-spec '{s}': iter must be non-negative"
                ));
            }
            return Ok(IterSpec::Range(None, Some(hi)));
        }
        if let Some((a, b)) = s.split_once('-') {
            let lo: i32 = a
                .parse()
                .map_err(|_| format!("invalid iter-spec '{s}': '{a}' is not an integer"))?;
            if lo < 0 {
                return Err(format!(
                    "invalid iter-spec '{s}': iter must be non-negative"
                ));
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
                return Err(format!(
                    "invalid iter-spec '{s}': iter must be non-negative"
                ));
            }
            if hi < lo {
                return Err(format!(
                    "invalid iter-spec '{s}': end ({hi}) is below start ({lo})"
                ));
            }
            return Ok(IterSpec::Range(Some(lo), Some(hi)));
        }
        // Bare "N"
        let n: i32 = s.parse().map_err(|_| {
            format!("invalid iter-spec '{s}': expected 'all', 'N', 'N-M', 'N-', or '-M'")
        })?;
        if n < 0 {
            return Err(format!(
                "invalid iter-spec '{s}': iter must be non-negative"
            ));
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
            other => Err(format!("unknown dump format '{other}' (expected: jsonl)")),
        }
    }
}

/// Payload-detail variant for the `iterate` dump category.
///
/// Iterate trajectories come in two sizes. `Summary` is small and
/// always cheap (m bits of active-set bitmap + a handful of scalars
/// per iteration); `Full` adds the full `x` and `slack` vectors per
/// iteration, which is the studio-grade payload but can run into
/// hundreds of MB on large problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IterateVariant {
    #[default]
    Summary,
    Full,
}

impl IterateVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            IterateVariant::Summary => "summary",
            IterateVariant::Full => "full",
        }
    }
}

/// Payload-detail variant for the `kkt` dump category.
///
/// `KOnly` (the default) emits only the K matrix and the solve's
/// RHS/solution. `WithLPattern` additionally emits the LDLᵀ factor's
/// strict-lower nonzero pattern (`L_irn` / `L_jcn`) and the fill-
/// reducing permutation `perm`. `WithLValues` further adds `L_vals`
/// in the same order as the pattern.
///
/// The L fields are emitted in *permuted* coordinates — the column /
/// row indices reference the permuted system K' = Pᵀ K P, and the
/// `perm` array carries the mapping back to original-variable space
/// (`perm[k] = original_row` for the k-th permuted row).
///
/// Backends that don't expose the factor pattern (e.g. MA57) silently
/// skip the L fields even when this variant requests them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KktVariant {
    #[default]
    KOnly,
    WithLPattern,
    WithLValues,
}

impl KktVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            KktVariant::KOnly => "k-only",
            KktVariant::WithLPattern => "with-l-pattern",
            KktVariant::WithLValues => "with-l-values",
        }
    }

    /// True if the variant asks for the L pattern (with or without
    /// values). Used by the dump site to short-circuit the
    /// `factor_pattern()` call when only K is wanted.
    pub fn wants_l_pattern(self) -> bool {
        matches!(self, KktVariant::WithLPattern | KktVariant::WithLValues)
    }

    /// True if the variant asks for the L numerical values.
    pub fn wants_l_values(self) -> bool {
        matches!(self, KktVariant::WithLValues)
    }
}

/// Parse the `kkt:` spec grammar — `[<iter-filter>][+L][+Lvals]`.
///
/// Recognised forms:
///
/// - empty / `all` / `5` / `5-10` / `5-` / `-10` → corresponding
///   filter, K-only (no L pattern).
/// - `<filter>+L` → K + LDLᵀ pattern + permutation.
/// - `<filter>+L+Lvals` → K + L pattern + L values.
/// - bare `+L` / `+L+Lvals` → all iters with the requested variant.
///
/// The suffixes are stripped *right-to-left* so the order
/// `<filter>+L+Lvals` is the only accepted spelling for values; the
/// reverse (`+Lvals+L`) is not recognised. `+Lvals` without `+L` is
/// also rejected — values without a pattern is meaningless.
pub fn parse_kkt_spec(s: &str) -> Result<(IterSpec, KktVariant), String> {
    let s = s.trim();
    // Strip `+Lvals` first (rightmost suffix), then `+L`. The order
    // matters: `+L+Lvals` should parse as both, `+L` alone as pattern-
    // only, `+Lvals` alone as an error.
    let (rest, has_lvals) = match s.strip_suffix("+Lvals") {
        Some(r) => (r, true),
        None => (s, false),
    };
    let (filter_str, has_l) = match rest.strip_suffix("+L") {
        Some(r) => (r, true),
        None => (rest, false),
    };
    if has_lvals && !has_l {
        return Err(format!(
            "invalid kkt-spec '{s}': '+Lvals' requires '+L' (use '+L+Lvals' for L pattern with values)"
        ));
    }
    let variant = if has_lvals {
        KktVariant::WithLValues
    } else if has_l {
        KktVariant::WithLPattern
    } else {
        KktVariant::KOnly
    };
    let filter_str = if filter_str.is_empty() {
        "all"
    } else {
        filter_str
    };
    let filter = IterSpec::parse(filter_str)?;
    Ok((filter, variant))
}

/// Parse the `iterate:` spec grammar — `[<iter-filter>[:<variant>]]`.
///
/// Recognised forms:
///
/// - empty / `all` / `5` / `5-10` / `5-` / `-10` → corresponding
///   filter, variant defaults to `summary`.
/// - `summary` / `full` → all iters, named variant.
/// - `<filter>:summary` / `<filter>:full` → both.
///
/// This is the only `DiagCategory` whose spec carries more than an
/// iter filter. Keeping the parser local to the iterate site avoids
/// growing every category's grammar.
pub fn parse_iterate_spec(s: &str) -> Result<(IterSpec, IterateVariant), String> {
    let s = s.trim();
    // Bare `summary` / `full` (no filter portion).
    if s == "summary" {
        return Ok((IterSpec::All, IterateVariant::Summary));
    }
    if s == "full" {
        return Ok((IterSpec::All, IterateVariant::Full));
    }
    // `<filter>:summary` / `<filter>:full`.
    let (filter_str, variant) = if let Some(rest) = s.strip_suffix(":summary") {
        (rest, IterateVariant::Summary)
    } else if let Some(rest) = s.strip_suffix(":full") {
        (rest, IterateVariant::Full)
    } else {
        (s, IterateVariant::Summary)
    };
    let filter_str = if filter_str.is_empty() {
        "all"
    } else {
        filter_str
    };
    let filter = IterSpec::parse(filter_str)?;
    Ok((filter, variant))
}

/// Static configuration: where to dump, in what format, with what
/// per-category iter filters. Constructed by the CLI, held by the
/// application, frozen for the duration of a solve.
#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    pub dump_dir: PathBuf,
    pub format: DumpFormat,
    pub categories: HashMap<DiagCategory, IterSpec>,
    /// Payload-detail for `DiagCategory::Iterate`. Only consulted
    /// when `Iterate` is in `categories`.
    pub iterate_variant: IterateVariant,
    /// Payload-detail for `DiagCategory::Kkt`. Only consulted when
    /// `Kkt` is in `categories`.
    pub kkt_variant: KktVariant,
}

impl DiagnosticsConfig {
    pub fn new(dump_dir: PathBuf) -> Self {
        Self {
            dump_dir,
            format: DumpFormat::Jsonl,
            categories: HashMap::new(),
            iterate_variant: IterateVariant::Summary,
            kkt_variant: KktVariant::KOnly,
        }
    }

    pub fn with_category(mut self, cat: DiagCategory, spec: IterSpec) -> Self {
        self.categories.insert(cat, spec);
        self
    }

    pub fn with_iterate_variant(mut self, v: IterateVariant) -> Self {
        self.iterate_variant = v;
        self
    }

    pub fn with_kkt_variant(mut self, v: KktVariant) -> Self {
        self.kkt_variant = v;
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
    /// Lazily-opened, persistent writer for the top-level
    /// `iterates.jsonl` stream. Opened on the first iterate emit and
    /// kept open across iterations (and through resto) so each
    /// outer/inner iteration appends one line in order.
    iterates_writer: RefCell<Option<BufWriter<fs::File>>>,
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
            iterates_writer: RefCell::new(None),
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

    /// True if the solver is currently inside a restoration sub-IPM
    /// run. Public, side-effect-free probe for emitters that need to
    /// tag rows with the restoration flag without mkdir-ing the iter
    /// directory (which `iter_dir` does).
    pub fn in_restoration(&self) -> bool {
        self.in_restoration.load(Ordering::SeqCst)
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
            self.config
                .dump_dir
                .join(format!("resto/parent_iter_{parent:03}/iter_{inner:03}"))
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

    /// Append one JSONL line to the persistent top-level
    /// `iterates.jsonl` stream, opening the file on first use. The
    /// writer is held across iterations so the emitter doesn't
    /// re-open the file every step (which would also truncate it).
    ///
    /// Caller supplies the already-encoded JSON record without a
    /// trailing newline; this method appends `\n` and flushes the
    /// buffer so a `SIGKILL`'d solve still leaves a parseable
    /// partial trace.
    pub fn append_iterate_line(&self, json: &str) -> std::io::Result<()> {
        let mut slot = self.iterates_writer.borrow_mut();
        if slot.is_none() {
            let path = self.config.dump_dir.join("iterates.jsonl");
            let f = fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(path)?;
            *slot = Some(BufWriter::new(f));
        }
        let w = slot.as_mut().expect("just initialized");
        w.write_all(json.as_bytes())?;
        w.write_all(b"\n")?;
        w.flush()
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
    fn iterate_spec_parses_all_combinations() {
        // Bare variant words: "all" filter, named variant.
        assert_eq!(
            parse_iterate_spec("summary").unwrap(),
            (IterSpec::All, IterateVariant::Summary)
        );
        assert_eq!(
            parse_iterate_spec("full").unwrap(),
            (IterSpec::All, IterateVariant::Full)
        );
        // Plain filter: defaults variant to Summary.
        assert_eq!(
            parse_iterate_spec("all").unwrap(),
            (IterSpec::All, IterateVariant::Summary)
        );
        assert_eq!(
            parse_iterate_spec("5").unwrap(),
            (IterSpec::Single(5), IterateVariant::Summary)
        );
        assert_eq!(
            parse_iterate_spec("5-10").unwrap(),
            (IterSpec::Range(Some(5), Some(10)), IterateVariant::Summary)
        );
        // Filter + variant.
        assert_eq!(
            parse_iterate_spec("all:summary").unwrap(),
            (IterSpec::All, IterateVariant::Summary)
        );
        assert_eq!(
            parse_iterate_spec("all:full").unwrap(),
            (IterSpec::All, IterateVariant::Full)
        );
        assert_eq!(
            parse_iterate_spec("5-:full").unwrap(),
            (IterSpec::Range(Some(5), None), IterateVariant::Full)
        );
        assert_eq!(
            parse_iterate_spec("10-20:full").unwrap(),
            (IterSpec::Range(Some(10), Some(20)), IterateVariant::Full)
        );
    }

    #[test]
    fn append_iterate_line_streams_rows_to_top_level() {
        let tmp = tempdir();
        let cfg =
            DiagnosticsConfig::new(tmp.clone()).with_category(DiagCategory::Iterate, IterSpec::All);
        let state = DiagnosticsState::new(cfg).unwrap();
        state.append_iterate_line("{\"iter\":0}").unwrap();
        state.append_iterate_line("{\"iter\":1}").unwrap();
        // Resto rows live inline in the same stream — the writer
        // doesn't care about the iter-dir nesting that other dump
        // sites use.
        state.enter_restoration();
        state
            .append_iterate_line("{\"iter\":0,\"restoration\":true}")
            .unwrap();
        state.exit_restoration();
        state.append_iterate_line("{\"iter\":2}").unwrap();

        let path = tmp.join("iterates.jsonl");
        let contents = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "{\"iter\":0}");
        assert_eq!(lines[2], "{\"iter\":0,\"restoration\":true}");
        fs::remove_dir_all(tmp).ok();
    }

    #[test]
    fn kkt_spec_parses_all_combinations() {
        // Empty / bare filter → K-only.
        assert_eq!(
            parse_kkt_spec("").unwrap(),
            (IterSpec::All, KktVariant::KOnly)
        );
        assert_eq!(
            parse_kkt_spec("all").unwrap(),
            (IterSpec::All, KktVariant::KOnly)
        );
        assert_eq!(
            parse_kkt_spec("5-10").unwrap(),
            (IterSpec::Range(Some(5), Some(10)), KktVariant::KOnly)
        );
        // +L pattern only.
        assert_eq!(
            parse_kkt_spec("+L").unwrap(),
            (IterSpec::All, KktVariant::WithLPattern)
        );
        assert_eq!(
            parse_kkt_spec("5-10+L").unwrap(),
            (IterSpec::Range(Some(5), Some(10)), KktVariant::WithLPattern)
        );
        assert_eq!(
            parse_kkt_spec("3+L").unwrap(),
            (IterSpec::Single(3), KktVariant::WithLPattern)
        );
        // +L+Lvals.
        assert_eq!(
            parse_kkt_spec("+L+Lvals").unwrap(),
            (IterSpec::All, KktVariant::WithLValues)
        );
        assert_eq!(
            parse_kkt_spec("5-10+L+Lvals").unwrap(),
            (IterSpec::Range(Some(5), Some(10)), KktVariant::WithLValues)
        );
    }

    #[test]
    fn kkt_spec_rejects_lvals_without_l() {
        assert!(parse_kkt_spec("+Lvals").is_err());
        assert!(parse_kkt_spec("5-10+Lvals").is_err());
    }

    #[test]
    fn iterate_spec_rejects_garbage_and_unknown_variants() {
        // Unknown variant after the colon: the parser strips no
        // suffix, falls back to whole-string filter parsing, which
        // then fails because "5-:bogus" is not a valid iter spec.
        assert!(parse_iterate_spec("5-:bogus").is_err());
        assert!(parse_iterate_spec("abc").is_err());
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
        let cfg =
            DiagnosticsConfig::new(tmp.clone()).with_category(DiagCategory::Kkt, IterSpec::All);
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
        let cfg =
            DiagnosticsConfig::new(tmp.clone()).with_category(DiagCategory::Kkt, IterSpec::All);
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
