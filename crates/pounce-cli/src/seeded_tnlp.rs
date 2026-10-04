//! Re-export of [`pounce_nlp::seeded_tnlp`], which moved down a crate so the
//! algorithm layer can re-seed its own re-scale retry (gh#983 item 3).
pub use pounce_nlp::seeded_tnlp::SeededTnlp;
