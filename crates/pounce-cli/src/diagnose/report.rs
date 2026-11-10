//! The findings schema `pounce diagnose` reports in (`pounce.diagnose/v1`).
//!
//! A finding is one observation about the model, with what it rests on kept
//! separate from how bad it is:
//!
//! * [`Severity`] — how much the user should care: `error` (the model as
//!   written cannot be solved, or a solve from this point will abort),
//!   `warning` (likely trouble), `info` (worth knowing).
//! * [`Basis`] — what kind of evidence backs it: a `proof` from the
//!   represented algebra and bounds, a `structural` fact about the sparsity
//!   pattern, a `numerical` observation at one named point and tolerance, or
//!   a `heuristic` pattern match.
//! * [`Confidence`] — how sure the check is that the observation is real.
//!
//! The two are deliberately independent: a numerical rank loss at one point
//! can be an `error` for the solve from that point and still only a
//! `numerical` observation about the model.
//!
//! A check that could not run is not silently absent. It is recorded in
//! [`Report::checks`] with status `unavailable` and a reason, so an empty
//! findings list can be told apart from "nothing was checked".

use pounce_common::types::Number;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// How much the user should care about a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

/// What kind of evidence a finding rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Follows from the represented algebra and bounds alone; holds at
    /// every point.
    Proof,
    /// Follows from the sparsity pattern alone. A structural rank loss is a
    /// bound on the numerical rank at every point.
    Structural,
    /// Observed at one named point under a stated tolerance. Says nothing
    /// by itself about any other point.
    Numerical,
    /// A pattern that often signals trouble but proves nothing.
    Heuristic,
}

impl Basis {
    pub fn as_str(self) -> &'static str {
        match self {
            Basis::Proof => "proof",
            Basis::Structural => "structural",
            Basis::Numerical => "numerical",
            Basis::Heuristic => "heuristic",
        }
    }
}

/// How sure the check is that the observation is real.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    Low,
    Medium,
    High,
    Certain,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Confidence::Low => "low",
            Confidence::Medium => "medium",
            Confidence::High => "high",
            Confidence::Certain => "certain",
        }
    }
}

/// One finding. Build with [`Finding::new`] and the chained setters.
#[derive(Debug, Clone)]
pub struct Finding {
    /// Stable identifier, `snake_case`. Codes are part of the schema:
    /// renaming one is a breaking change for anyone filtering on it.
    pub code: &'static str,
    /// The check that produced it (one of [`Report::checks`]).
    pub check: &'static str,
    pub severity: Severity,
    pub basis: Basis,
    pub confidence: Confidence,
    /// What was seen, in one or two sentences, naming the entities.
    pub observation: String,
    /// Why the solver or the user should care.
    pub why_it_matters: String,
    /// The cheapest next thing to look at. An investigation step, never an
    /// instruction to change the model.
    pub suggested_next_check: String,
    /// Variable names involved.
    pub variables: Vec<String>,
    /// Constraint names involved (`objective` for the objective).
    pub constraints: Vec<String>,
    /// Supporting numbers: intervals, values, tolerances, the point label.
    pub evidence: Value,
}

impl Finding {
    pub fn new(
        check: &'static str,
        code: &'static str,
        severity: Severity,
        basis: Basis,
        confidence: Confidence,
        observation: impl Into<String>,
    ) -> Self {
        Finding {
            code,
            check,
            severity,
            basis,
            confidence,
            observation: observation.into(),
            why_it_matters: String::new(),
            suggested_next_check: String::new(),
            variables: Vec::new(),
            constraints: Vec::new(),
            evidence: Value::Null,
        }
    }

    pub fn why(mut self, s: impl Into<String>) -> Self {
        self.why_it_matters = s.into();
        self
    }

    pub fn next(mut self, s: impl Into<String>) -> Self {
        self.suggested_next_check = s.into();
        self
    }

    pub fn vars(mut self, v: Vec<String>) -> Self {
        self.variables = v;
        self
    }

    pub fn cons(mut self, c: Vec<String>) -> Self {
        self.constraints = c;
        self
    }

    pub fn evidence(mut self, e: Value) -> Self {
        self.evidence = e;
        self
    }

    fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "check": self.check,
            "severity": self.severity.as_str(),
            "basis": self.basis.as_str(),
            "confidence": self.confidence.as_str(),
            "observation": self.observation,
            "why_it_matters": self.why_it_matters,
            "suggested_next_check": self.suggested_next_check,
            "affected": {
                "variables": self.variables,
                "constraints": self.constraints,
            },
            "evidence": self.evidence,
        })
    }
}

/// Whether a check ran.
#[derive(Debug, Clone)]
pub enum CheckStatus {
    Ran,
    /// The check did not run; the string says why.
    Unavailable(String),
}

/// One entry of [`Report::checks`].
#[derive(Debug, Clone)]
pub struct CheckRecord {
    pub name: &'static str,
    pub status: CheckStatus,
}

/// Collects findings for one check, capping how many of each code are kept
/// so a model with ten thousand `log(x)` rows does not produce a
/// ten-thousand-entry report. The cap is per code; the number dropped is
/// reported in [`Report::truncated`].
pub struct Sink<'a> {
    pub check: &'static str,
    report: &'a mut Report,
}

impl Sink<'_> {
    pub fn push(&mut self, f: Finding) {
        debug_assert_eq!(f.check, self.check, "finding filed under the wrong check");
        let kept = self.report.kept_per_code.entry(f.code).or_insert(0);
        if *kept < self.report.max_per_code {
            *kept += 1;
            self.report.findings.push(f);
        } else {
            *self.report.truncated.entry(f.code).or_insert(0) += 1;
            *self.report.dropped.entry(f.severity).or_insert(0) += 1;
        }
    }
}

/// The whole report.
#[derive(Debug, Clone)]
pub struct Report {
    pub source: String,
    pub sha256: Option<String>,
    pub n_vars: usize,
    pub n_cons: usize,
    /// Label of the point the numerical checks evaluated at.
    pub point: String,
    pub checks: Vec<CheckRecord>,
    pub findings: Vec<Finding>,
    /// Per-code count of findings dropped by the cap.
    pub truncated: BTreeMap<&'static str, usize>,
    /// Findings dropped by the cap, per severity. A code can mix severities
    /// (a non-finite value the clamp clears is a warning, one it does not is
    /// an error), so the per-code count alone cannot say which were dropped.
    dropped: BTreeMap<Severity, usize>,
    kept_per_code: BTreeMap<&'static str, usize>,
    pub max_per_code: usize,
    /// Tolerances used, by name, so a reader can reproduce the run.
    pub tolerances: BTreeMap<&'static str, Number>,
    /// Extra structured sections (the embedded check-x0 report, summary
    /// statistics), keyed by section name.
    pub sections: BTreeMap<&'static str, Value>,
}

impl Report {
    pub fn new(source: String, n_vars: usize, n_cons: usize, max_per_code: usize) -> Self {
        Report {
            source,
            sha256: None,
            n_vars,
            n_cons,
            point: String::new(),
            checks: Vec::new(),
            findings: Vec::new(),
            truncated: BTreeMap::new(),
            dropped: BTreeMap::new(),
            kept_per_code: BTreeMap::new(),
            max_per_code: max_per_code.max(1),
            tolerances: BTreeMap::new(),
            sections: BTreeMap::new(),
        }
    }

    /// Record that `check` ran and return a sink for its findings.
    pub fn ran(&mut self, check: &'static str) -> Sink<'_> {
        self.checks.push(CheckRecord {
            name: check,
            status: CheckStatus::Ran,
        });
        Sink {
            check,
            report: self,
        }
    }

    /// Record that `check` could not run, and why.
    pub fn unavailable(&mut self, check: &'static str, reason: impl Into<String>) {
        self.checks.push(CheckRecord {
            name: check,
            status: CheckStatus::Unavailable(reason.into()),
        });
    }

    /// Findings of severity `sev`, counting those the cap dropped.
    pub fn count(&self, sev: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity == sev).count()
            + self.dropped.get(&sev).copied().unwrap_or(0)
    }

    /// `true` when any finding is an error, including ones the cap dropped.
    pub fn has_errors(&self) -> bool {
        self.count(Severity::Error) > 0
    }

    /// `ERRORS` / `WARNINGS` / `CLEAN`.
    pub fn verdict(&self) -> &'static str {
        if self.has_errors() {
            "ERRORS"
        } else if self.count(Severity::Warning) > 0 {
            "WARNINGS"
        } else {
            "CLEAN"
        }
    }

    /// Findings sorted most severe first, stable within a severity (checks
    /// run in a fixed order, so the order is deterministic).
    pub fn sorted(&self) -> Vec<&Finding> {
        let mut v: Vec<&Finding> = self.findings.iter().collect();
        v.sort_by_key(|f| std::cmp::Reverse(f.severity));
        v
    }

    pub fn to_json(&self) -> Value {
        let checks: Vec<Value> = self
            .checks
            .iter()
            .map(|c| match &c.status {
                CheckStatus::Ran => json!({"check": c.name, "status": "ran"}),
                CheckStatus::Unavailable(r) => {
                    json!({"check": c.name, "status": "unavailable", "reason": r})
                }
            })
            .collect();
        let findings: Vec<Value> = self.sorted().iter().map(|f| f.to_json()).collect();
        let mut sections = serde_json::Map::new();
        for (k, v) in &self.sections {
            sections.insert((*k).to_string(), v.clone());
        }
        json!({
            "schema": "pounce.diagnose/v1",
            "solver": format!("pounce {}", env!("CARGO_PKG_VERSION")),
            "problem": {
                "source": self.source,
                "sha256": self.sha256,
                "n_vars": self.n_vars,
                "n_cons": self.n_cons,
            },
            "point": self.point,
            "tolerances": self.tolerances,
            "verdict": self.verdict(),
            "summary": {
                "errors": self.count(Severity::Error),
                "warnings": self.count(Severity::Warning),
                "info": self.count(Severity::Info),
            },
            "checks": checks,
            "findings": findings,
            "truncated": self.truncated,
            "sections": Value::Object(sections),
        })
    }

    /// The human-readable report. Errors and warnings are listed in full;
    /// info findings are summarized per code after the first few, so the
    /// signal is not buried. The JSON carries everything.
    pub fn to_text(&self) -> String {
        use std::fmt::Write;
        const INFO_SHOWN_PER_CODE: usize = 3;
        let mut out = String::new();
        let _ = writeln!(out, "pounce diagnose: {}", self.source);
        let _ = writeln!(
            out,
            "  {} variables, {} constraints; numerical checks at {}",
            self.n_vars, self.n_cons, self.point
        );
        let _ = writeln!(
            out,
            "  verdict: {}  ({} error(s), {} warning(s), {} info)",
            self.verdict(),
            self.count(Severity::Error),
            self.count(Severity::Warning),
            self.count(Severity::Info)
        );
        let mut info_shown: BTreeMap<&str, usize> = BTreeMap::new();
        let mut info_hidden: BTreeMap<&str, usize> = BTreeMap::new();
        for f in self.sorted() {
            if f.severity == Severity::Info {
                let shown = info_shown.entry(f.code).or_insert(0);
                if *shown >= INFO_SHOWN_PER_CODE {
                    *info_hidden.entry(f.code).or_insert(0) += 1;
                    continue;
                }
                *shown += 1;
            }
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "  [{}] {}  ({}, {} confidence)",
                f.severity.as_str().to_uppercase(),
                f.code,
                f.basis.as_str(),
                f.confidence.as_str()
            );
            let _ = writeln!(out, "    {}", f.observation);
            if !f.why_it_matters.is_empty() {
                let _ = writeln!(out, "    why: {}", f.why_it_matters);
            }
            if !f.suggested_next_check.is_empty() {
                let _ = writeln!(out, "    next: {}", f.suggested_next_check);
            }
        }
        if !info_hidden.is_empty() || !self.truncated.is_empty() {
            let _ = writeln!(out);
            for (code, n) in &info_hidden {
                let _ = writeln!(out, "  ... {n} more info `{code}` finding(s) (see --json)");
            }
            for (code, n) in &self.truncated {
                let _ = writeln!(
                    out,
                    "  ... {n} more `{code}` finding(s) dropped by --max-per-code {}",
                    self.max_per_code
                );
            }
        }
        let unavailable: Vec<&CheckRecord> = self
            .checks
            .iter()
            .filter(|c| matches!(c.status, CheckStatus::Unavailable(_)))
            .collect();
        if !unavailable.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "  checks that did not run:");
            for c in unavailable {
                if let CheckStatus::Unavailable(r) = &c.status {
                    let _ = writeln!(out, "    {}: {r}", c.name);
                }
            }
        }
        out
    }
}

/// A number for JSON evidence: finite values as numbers, non-finite as the
/// strings `"inf"`, `"-inf"`, `"nan"` (JSON has no spelling for them).
pub fn num(v: Number) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        json!(v.to_string())
    }
}

/// A number for prose: plain decimal between 1e-4 and 1e6, scientific
/// otherwise, and no 17-digit tails. `inf`/`-inf`/`NaN` as Rust prints them.
pub fn fmt_num(v: Number) -> String {
    if !v.is_finite() {
        return v.to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if (1e-4..1e6).contains(&a) {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    } else {
        format!("{v:.3e}")
    }
}

/// An interval for JSON evidence.
pub fn interval(lo: Number, hi: Number) -> Value {
    json!([num(lo), num(hi)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_error_dropped_by_the_cap_still_sets_the_verdict() {
        let mut r = Report::new("t".into(), 0, 0, 1);
        let mut sink = r.ran("c");
        let f = |sev| Finding::new("c", "mixed", sev, Basis::Numerical, Confidence::High, "x");
        sink.push(f(Severity::Warning));
        sink.push(f(Severity::Error)); // dropped: the cap is 1 per code
        assert!(r.findings.iter().all(|f| f.severity == Severity::Warning));
        assert!(r.has_errors());
        assert_eq!(r.verdict(), "ERRORS");
        assert_eq!(r.count(Severity::Error), 1);
        assert_eq!(r.count(Severity::Warning), 1);
    }
}
