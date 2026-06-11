//! Citation lister for `pounce --cite` (pounce#…): tells a user which
//! papers / software to cite when they publish results obtained with
//! pounce.
//!
//! Two tiers, per the agreed design:
//!   * **static core** — always: pounce itself + Wächter-Biegler.
//!   * **solve-aware extras** — when a solve-report JSON is supplied,
//!     add papers for features the run actually used. In v1 the only
//!     such signal carried by the report schema is the restoration
//!     phase (`statistics.restoration_calls > 0`).
//!
//! Citation data is reused from `pounce_studio_core::glossary` (the same
//! curated table that backs `pounce-studio citations` and the MCP tool),
//! so there is a single source of truth rather than a CLI-local copy.

use crate::solve_report::SolveReport;
use pounce_studio_core::glossary::{citation_by_key, solve_feature_keys, Citation};

/// A selected citation together with the short reason it was included,
/// shown to the user as a "why cited" note in the human renderer.
pub struct Selected {
    pub citation: &'static Citation,
    pub reason: &'static str,
}

/// Build the ordered, de-duplicated list of citations for this run.
///
/// `report` is the parsed `--cite <report.json>` if one was given.
/// The static core is always first; solve-aware extras follow in the
/// order their features were detected.
pub fn select(report: Option<&SolveReport>) -> Vec<Selected> {
    let mut out: Vec<Selected> = Vec::new();
    let mut seen: Vec<&'static str> = Vec::new();

    let push = |feature: &str,
                reason: &'static str,
                out: &mut Vec<Selected>,
                seen: &mut Vec<&'static str>| {
        let Some(keys) = solve_feature_keys(feature) else {
            return;
        };
        for &k in keys {
            if seen.contains(&k) {
                continue;
            }
            if let Some(citation) = citation_by_key(k) {
                seen.push(k);
                out.push(Selected { citation, reason });
            }
        }
    };

    // Static core — every pounce run should cite these.
    push(
        "core",
        "core solver (cite for any pounce result)",
        &mut out,
        &mut seen,
    );

    // Solve-aware extras.
    if let Some(r) = report {
        if r.statistics.restoration_calls > 0 {
            push(
                "restoration",
                "the restoration phase was entered during this solve",
                &mut out,
                &mut seen,
            );
        }
    }

    out
}

/// Human-readable rendering: a short header followed by one stanza per
/// citation with title, author, year, DOI, and the "why cited" note.
pub fn render_human(selected: &[Selected]) -> String {
    let mut s = String::new();
    s.push_str("Please cite the following when publishing results obtained with pounce:\n");
    for (i, sel) in selected.iter().enumerate() {
        let c = sel.citation;
        s.push_str(&format!("\n[{}] {}\n", i + 1, c.title));
        s.push_str(&format!("    {} ({})\n", c.author, c.year));
        if !c.venue.is_empty() {
            s.push_str(&format!("    {}\n", c.venue));
        }
        if !c.doi.is_empty() {
            s.push_str(&format!("    doi:{}\n", c.doi));
        }
        s.push_str(&format!("    — {}\n", sel.reason));
    }
    s
}

/// BibTeX rendering: one entry per citation, ready to paste into a
/// `.bib` file. Fields are emitted only when non-empty.
pub fn render_bibtex(selected: &[Selected]) -> String {
    let mut s = String::new();
    for (i, sel) in selected.iter().enumerate() {
        let c = sel.citation;
        if i > 0 {
            s.push('\n');
        }
        s.push_str(&format!("@{}{{{},\n", c.entry_type, c.key));
        s.push_str(&format!("  title = {{{}}},\n", c.title));
        if !c.author.is_empty() {
            s.push_str(&format!("  author = {{{}}},\n", c.author));
        }
        if !c.year.is_empty() {
            s.push_str(&format!("  year = {{{}}},\n", c.year));
        }
        if !c.venue.is_empty() {
            // `@article` expects `journal`; `howpublished` is an `@misc` field
            // many styles ignore, which would silently drop a journal venue.
            let venue_field = if c.entry_type == "article" {
                "journal"
            } else {
                "howpublished"
            };
            s.push_str(&format!("  {venue_field} = {{{}}},\n", c.venue));
        }
        if !c.doi.is_empty() {
            s.push_str(&format!("  doi = {{{}}},\n", c.doi));
        }
        s.push_str("}\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve_report::{InputDescriptor, ReportBuilder, ReportDetail};

    fn report(restoration_calls: i32) -> SolveReport {
        let mut r = ReportBuilder::new(
            ReportDetail::Summary,
            InputDescriptor::Builtin {
                name: "test".into(),
            },
        )
        .finish();
        r.statistics.restoration_calls = restoration_calls;
        r
    }

    #[test]
    fn core_always_present_without_report() {
        let sel = select(None);
        let keys: Vec<_> = sel.iter().map(|s| s.citation.key).collect();
        assert!(keys.contains(&"pounce2026"));
        assert!(keys.contains(&"wachter2006"));
        // No solve report → no restoration paper.
        assert!(!keys.contains(&"byrd2010"));
    }

    #[test]
    fn restoration_adds_byrd_when_report_shows_it() {
        let r = report(3);
        let sel = select(Some(&r));
        let keys: Vec<_> = sel.iter().map(|s| s.citation.key).collect();
        assert!(keys.contains(&"pounce2026"));
        assert!(keys.contains(&"byrd2010"));
    }

    #[test]
    fn no_restoration_no_byrd() {
        let r = report(0);
        let sel = select(Some(&r));
        let keys: Vec<_> = sel.iter().map(|s| s.citation.key).collect();
        assert!(!keys.contains(&"byrd2010"));
    }

    #[test]
    fn human_render_mentions_pounce_and_doi() {
        let out = render_human(&select(None));
        assert!(out.contains("POUNCE"));
        assert!(out.contains("10.5281/zenodo.20387011"));
    }

    #[test]
    fn bibtex_render_is_pasteable() {
        let out = render_bibtex(&select(None));
        assert!(out.contains("@software{pounce2026,"));
        assert!(out.contains("@article{wachter2006,"));
        assert!(out.contains("doi = {10.5281/zenodo.20387011}"));
    }
}
