//! Logging / journaling.
//!
//! Mirrors `Common/IpJournalist.{hpp,cpp}`. The Journalist owns
//! multiple `Journal`s (file/stream sinks); each Journal has a
//! per-category print level. A message of (level, category) is sent
//! to a Journal iff `level <= journal.print_level[category]`
//! (`J_INSUPPRESSIBLE = -1` always passes).
//!
//! Public API names match upstream as closely as Rust idioms allow.
//! Iteration log diffing in Phase 7 depends on byte-identical lines;
//! `printf!` semantics are replaced with direct write of pre-formatted
//! strings — callers will use `std::fmt`/`format!` to assemble the
//! line and pass the result through [`Journalist::print`].

use crate::types::Index;
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::Mutex;

/// Print level. Numeric values match Ipopt's `EJournalLevel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum JournalLevel {
    J_INSUPPRESSIBLE = -1,
    J_NONE = 0,
    J_ERROR = 1,
    J_STRONGWARNING = 2,
    J_SUMMARY = 3,
    J_WARNING = 4,
    J_ITERSUMMARY = 5,
    J_DETAILED = 6,
    J_MOREDETAILED = 7,
    J_VECTOR = 8,
    J_MOREVECTOR = 9,
    J_MATRIX = 10,
    J_MOREMATRIX = 11,
    J_ALL = 12,
}

impl JournalLevel {
    pub const J_LAST_LEVEL: i32 = 13;
}

/// Category. Numeric values match `EJournalCategory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum JournalCategory {
    J_DBG = 0,
    J_STATISTICS = 1,
    J_MAIN = 2,
    J_INITIALIZATION = 3,
    J_BARRIER_UPDATE = 4,
    J_SOLVE_PD_SYSTEM = 5,
    J_FRAC_TO_BOUND = 6,
    J_LINEAR_ALGEBRA = 7,
    J_LINE_SEARCH = 8,
    J_HESSIAN_APPROXIMATION = 9,
    J_SOLUTION = 10,
    J_DOCUMENTATION = 11,
    J_NLP = 12,
    J_TIMING_STATISTICS = 13,
    J_USER_APPLICATION = 14,
    J_USER1 = 15,
    J_USER2 = 16,
    J_USER3 = 17,
    J_USER4 = 18,
    J_USER5 = 19,
    J_USER6 = 20,
    J_USER7 = 21,
    J_USER8 = 22,
    J_USER9 = 23,
    J_USER10 = 24,
    J_USER11 = 25,
    J_USER12 = 26,
    J_USER13 = 27,
    J_USER14 = 28,
    J_USER15 = 29,
    J_USER16 = 30,
    J_USER17 = 31,
}

impl JournalCategory {
    pub const J_LAST_CATEGORY: usize = 32;
}

/// Trait for a single output sink. Implementors handle one of
/// stdout/stderr/a file/a string buffer. Mirrors `Ipopt::Journal`.
pub trait Journal: Send + Sync {
    fn name(&self) -> &str;

    /// Acceptance check — returns true iff the journal would emit a
    /// message at `(level, category)`.
    fn is_accepted(&self, category: JournalCategory, level: JournalLevel) -> bool;

    fn set_print_level(&self, category: JournalCategory, level: JournalLevel);

    fn set_all_print_levels(&self, level: JournalLevel);

    /// Emit a pre-formatted string (callers do their own formatting).
    fn print(&self, category: JournalCategory, level: JournalLevel, s: &str);

    fn flush(&self);
}

/// Per-category level table shared by every concrete Journal impl.
struct LevelTable {
    levels: [i32; JournalCategory::J_LAST_CATEGORY],
}

impl LevelTable {
    fn new(default_level: JournalLevel) -> Self {
        Self {
            levels: [default_level as i32; JournalCategory::J_LAST_CATEGORY],
        }
    }

    fn is_accepted(&self, category: JournalCategory, level: JournalLevel) -> bool {
        // J_INSUPPRESSIBLE always emits (matches upstream IsAccepted).
        if (level as i32) == JournalLevel::J_INSUPPRESSIBLE as i32 {
            return true;
        }
        (level as i32) <= self.levels[category as usize]
    }

    fn set_level(&mut self, category: JournalCategory, level: JournalLevel) {
        self.levels[category as usize] = level as i32;
    }

    fn set_all(&mut self, level: JournalLevel) {
        for v in &mut self.levels {
            *v = level as i32;
        }
    }
}

enum FileSink {
    Stdout,
    Stderr,
    File(File),
}

impl FileSink {
    fn write(&mut self, s: &str) -> io::Result<()> {
        match self {
            FileSink::Stdout => io::stdout().write_all(s.as_bytes()),
            FileSink::Stderr => io::stderr().write_all(s.as_bytes()),
            FileSink::File(f) => f.write_all(s.as_bytes()),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            FileSink::Stdout => io::stdout().flush(),
            FileSink::Stderr => io::stderr().flush(),
            FileSink::File(f) => f.flush(),
        }
    }
}

/// Mirrors `FileJournal` — writes to stdout/stderr/disk.
pub struct FileJournal {
    name: String,
    levels: Mutex<LevelTable>,
    sink: Mutex<FileSink>,
}

impl FileJournal {
    pub fn new(name: impl Into<String>, default_level: JournalLevel) -> Self {
        Self {
            name: name.into(),
            levels: Mutex::new(LevelTable::new(default_level)),
            sink: Mutex::new(FileSink::Stdout),
        }
    }

    /// Mirrors `FileJournal::Open`. `"stdout"`/`"stderr"` are
    /// recognised as special filenames. Returns false if the file
    /// could not be opened.
    pub fn open(&self, fname: &str, append: bool) -> bool {
        let new_sink = match fname {
            "stdout" => FileSink::Stdout,
            "stderr" => FileSink::Stderr,
            other => {
                let mut opts = OpenOptions::new();
                opts.write(true).create(true);
                if append {
                    opts.append(true);
                } else {
                    opts.truncate(true);
                }
                match opts.open(other) {
                    Ok(f) => FileSink::File(f),
                    Err(_) => return false,
                }
            }
        };
        match self.sink.lock() {
            Ok(mut s) => {
                *s = new_sink;
                true
            }
            _ => false,
        }
    }
}

impl Journal for FileJournal {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_accepted(&self, category: JournalCategory, level: JournalLevel) -> bool {
        self.levels
            .lock()
            .map(|t| t.is_accepted(category, level))
            .unwrap_or(false)
    }

    fn set_print_level(&self, category: JournalCategory, level: JournalLevel) {
        if let Ok(mut t) = self.levels.lock() {
            t.set_level(category, level);
        }
    }

    fn set_all_print_levels(&self, level: JournalLevel) {
        if let Ok(mut t) = self.levels.lock() {
            t.set_all(level);
        }
    }

    fn print(&self, category: JournalCategory, level: JournalLevel, s: &str) {
        if !self.is_accepted(category, level) {
            return;
        }
        if let Ok(mut sink) = self.sink.lock() {
            let _ = sink.write(s);
        }
    }

    fn flush(&self) {
        if let Ok(mut sink) = self.sink.lock() {
            let _ = sink.flush();
        }
    }
}

/// In-memory sink used by tests and the option-printing path.
pub struct StringJournal {
    name: String,
    levels: Mutex<LevelTable>,
    buffer: Mutex<String>,
}

impl StringJournal {
    pub fn new(name: impl Into<String>, default_level: JournalLevel) -> Self {
        Self {
            name: name.into(),
            levels: Mutex::new(LevelTable::new(default_level)),
            buffer: Mutex::new(String::new()),
        }
    }

    pub fn contents(&self) -> String {
        self.buffer.lock().map(|b| b.clone()).unwrap_or_default()
    }

    pub fn take(&self) -> String {
        self.buffer
            .lock()
            .map(|mut b| std::mem::take(&mut *b))
            .unwrap_or_default()
    }
}

impl Journal for StringJournal {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_accepted(&self, category: JournalCategory, level: JournalLevel) -> bool {
        self.levels
            .lock()
            .map(|t| t.is_accepted(category, level))
            .unwrap_or(false)
    }

    fn set_print_level(&self, category: JournalCategory, level: JournalLevel) {
        if let Ok(mut t) = self.levels.lock() {
            t.set_level(category, level);
        }
    }

    fn set_all_print_levels(&self, level: JournalLevel) {
        if let Ok(mut t) = self.levels.lock() {
            t.set_all(level);
        }
    }

    fn print(&self, category: JournalCategory, level: JournalLevel, s: &str) {
        if !self.is_accepted(category, level) {
            return;
        }
        if let Ok(mut buf) = self.buffer.lock() {
            buf.push_str(s);
        }
    }

    fn flush(&self) {}
}

/// The Journalist owns a list of journals and dispatches messages.
/// Mirrors `Ipopt::Journalist`.
#[derive(Default)]
pub struct Journalist {
    journals: RefCell<Vec<Arc<dyn Journal>>>,
}

impl Journalist {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_journal(&self, j: Arc<dyn Journal>) -> bool {
        match self.journals.try_borrow_mut() {
            Ok(journals) => {
                for existing in journals.iter() {
                    if existing.name() == j.name() {
                        return false;
                    }
                }
                drop(journals);
                self.journals.borrow_mut().push(j);
                true
            }
            _ => false,
        }
    }

    /// Convenience: add a `FileJournal` writing to `fname`.
    pub fn add_file_journal(
        &self,
        location_name: &str,
        fname: &str,
        default_level: JournalLevel,
        append: bool,
    ) -> Option<Arc<FileJournal>> {
        let j = Arc::new(FileJournal::new(location_name, default_level));
        if !j.open(fname, append) {
            return None;
        }
        let dyn_j: Arc<dyn Journal> = j.clone();
        if !self.add_journal(dyn_j) {
            return None;
        }
        Some(j)
    }

    pub fn get_journal(&self, location_name: &str) -> Option<Arc<dyn Journal>> {
        self.journals
            .borrow()
            .iter()
            .find(|j| j.name() == location_name)
            .cloned()
    }

    pub fn delete_all_journals(&self) {
        self.journals.borrow_mut().clear();
    }

    /// Emit a pre-formatted string to every accepting journal.
    /// Equivalent to `Journalist::Printf` after the C-style format
    /// expansion has been done in the caller.
    pub fn print(&self, level: JournalLevel, category: JournalCategory, s: &str) {
        for j in self.journals.borrow().iter() {
            j.print(category, level, s);
        }
    }

    /// Mirrors `PrintfIndented` — prepends `2 * indent_level` spaces.
    pub fn print_indented(
        &self,
        level: JournalLevel,
        category: JournalCategory,
        indent_level: Index,
        s: &str,
    ) {
        let pad = " ".repeat((indent_level.max(0) as usize) * 2);
        // Indent every line so multi-line payloads match upstream.
        let mut out = String::with_capacity(s.len() + pad.len());
        let mut first = true;
        for line in s.split_inclusive('\n') {
            if !first || !line.is_empty() {
                out.push_str(&pad);
            }
            out.push_str(line);
            first = false;
        }
        self.print(level, category, &out);
    }

    /// Mirrors `ProduceOutput` — true iff at least one journal accepts.
    pub fn produce_output(&self, level: JournalLevel, category: JournalCategory) -> bool {
        self.journals
            .borrow()
            .iter()
            .any(|j| j.is_accepted(category, level))
    }

    pub fn flush_buffer(&self) {
        for j in self.journals.borrow().iter() {
            j.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_filtering() {
        let jnlst = Journalist::new();
        let j = Arc::new(StringJournal::new("buf", JournalLevel::J_SUMMARY));
        jnlst.add_journal(j.clone());
        jnlst.print(JournalLevel::J_ERROR, JournalCategory::J_MAIN, "err\n");
        jnlst.print(
            JournalLevel::J_DETAILED,
            JournalCategory::J_MAIN,
            "detail\n",
        );
        let s = j.contents();
        assert!(s.contains("err"));
        assert!(!s.contains("detail"));
    }

    #[test]
    fn insuppressible_always_emits() {
        let jnlst = Journalist::new();
        let j = Arc::new(StringJournal::new("buf", JournalLevel::J_NONE));
        jnlst.add_journal(j.clone());
        jnlst.print(
            JournalLevel::J_INSUPPRESSIBLE,
            JournalCategory::J_MAIN,
            "x\n",
        );
        assert_eq!(j.contents(), "x\n");
    }

    #[test]
    fn produce_output_reflects_journals() {
        let jnlst = Journalist::new();
        assert!(!jnlst.produce_output(JournalLevel::J_ERROR, JournalCategory::J_MAIN));
        let j = Arc::new(StringJournal::new("buf", JournalLevel::J_SUMMARY));
        jnlst.add_journal(j);
        assert!(jnlst.produce_output(JournalLevel::J_ERROR, JournalCategory::J_MAIN));
        assert!(!jnlst.produce_output(JournalLevel::J_DETAILED, JournalCategory::J_MAIN));
    }

    #[test]
    fn indent_prepends_two_spaces_per_level() {
        let jnlst = Journalist::new();
        let j = Arc::new(StringJournal::new("buf", JournalLevel::J_ALL));
        jnlst.add_journal(j.clone());
        jnlst.print_indented(JournalLevel::J_SUMMARY, JournalCategory::J_MAIN, 2, "x\n");
        assert_eq!(j.contents(), "    x\n");
    }
}
