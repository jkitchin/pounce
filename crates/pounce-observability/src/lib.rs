//! Observability wiring for POUNCE (pounce#71).
//!
//! This crate owns the `tracing` subscriber install and the bridge
//! between the structured per-iteration event and the JSON solve
//! report. It is kept separate from the leaf `pounce-common` (which
//! holds the pure color palette) because the collector needs both
//! [`pounce_nlp::solve_statistics::IterRecord`] and
//! `tracing-subscriber`.
//!
//! ## Two output channels, one event
//!
//! Each Newton iteration emits a single structured event at
//! [`ITER_TARGET`] carrying `iter`, `mu`, `alpha_primal`, … The human
//! terminal never sees this event (the colored fixed-width table,
//! printed directly by `pounce-algorithm`, is its visual form, so the
//! text console layer filters `pounce::iteration` out). Machines get
//! it two ways:
//!
//! * `POUNCE_LOG_FORMAT=json` → the JSON layer prints it to stderr;
//! * the [`IterCollectorLayer`] rebuilds an `IterRecord` from its
//!   fields and appends it to the active [`IterCaptureGuard`] slot,
//!   which the application drains into the solve report.
//!
//! The collector skips events nested inside a `restoration` span, so
//! the report captures only the outer solve's iterations (including
//! `'R'`-marked outer iters) and not the restoration sub-solve's inner
//! IPM iterations — matching the pre-tracing behavior exactly.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::cell::RefCell;

use pounce_common::types::{Index, Number};
use pounce_nlp::solve_statistics::IterRecord;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Target of the structured per-iteration event. The text console
/// layer filters this target out (the colored table is its human
/// form); the JSON layer and [`IterCollectorLayer`] keep it.
pub const ITER_TARGET: &str = "pounce::iteration";

/// Span name whose presence in an event's ancestry marks the event as
/// belonging to the restoration sub-solve. The collector uses it to
/// exclude inner restoration iterations from the report.
pub const RESTORATION_SPAN: &str = "restoration";

// ---- Per-solve capture slot ----

thread_local! {
    /// Active capture buffer for the current solve, or `None` when no
    /// solve on this thread is recording its iteration history.
    static CAPTURE: RefCell<Option<Vec<IterRecord>>> = const { RefCell::new(None) };
}

/// Set once at subscriber install when `POUNCE_LOG_FORMAT=json`, so the
/// per-iteration event is emitted (for the JSON sink) even when no
/// in-process capture is active.
static JSON_LOGGING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the per-iteration `pounce::iteration` event has a consumer
/// right now, so the algorithm can skip emitting it (and the field
/// evaluation it entails) when nothing would observe it.
///
/// True when either an [`IterCaptureGuard`] is active on this thread
/// (the JSON report wants the trajectory) or JSON logging is installed
/// (the stderr sink wants it). In the common default run — text logs,
/// no iter-history capture — this is `false`, so the event costs
/// nothing.
pub fn iteration_event_wanted() -> bool {
    if JSON_LOGGING.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    CAPTURE.with(|c| c.borrow().is_some())
}

/// RAII activation of per-iteration capture for one solve.
///
/// Construct with [`IterCaptureGuard::start`] immediately before the
/// solve and call [`IterCaptureGuard::finish`] after it to take the
/// collected records. Solves run synchronously on one thread, so a
/// thread-local slot suffices; restoration sub-solves are excluded by
/// span scoping in the collector rather than by nesting guards.
#[must_use = "call finish() to retrieve the captured iteration history"]
pub struct IterCaptureGuard {
    /// Any buffer that was active before this guard, restored on drop
    /// so sequential or accidentally-nested solves don't clobber it.
    prev: Option<Vec<IterRecord>>,
}

impl IterCaptureGuard {
    /// Begin capturing iteration records on this thread.
    pub fn start() -> Self {
        let prev = CAPTURE.with(|c| c.borrow_mut().replace(Vec::new()));
        Self { prev }
    }

    /// End capture and return the records collected since [`start`].
    ///
    /// [`start`]: IterCaptureGuard::start
    pub fn finish(mut self) -> Vec<IterRecord> {
        let prev = self.prev.take();
        let captured = CAPTURE
            .with(|c| std::mem::replace(&mut *c.borrow_mut(), prev))
            .unwrap_or_default();
        // Skip `Drop`: it would re-restore `self.prev` (now `None`) and
        // clobber the buffer we just put back for an enclosing guard.
        std::mem::forget(self);
        captured
    }
}

impl Drop for IterCaptureGuard {
    fn drop(&mut self) {
        // Restore the previous buffer if `finish` wasn't called.
        let prev = self.prev.take();
        CAPTURE.with(|c| *c.borrow_mut() = prev);
    }
}

/// Append a record to the active capture slot, if any.
fn push_record(rec: IterRecord) {
    CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push(rec);
        }
    });
}

// ---- Event → IterRecord visitor ----

#[derive(Default)]
struct IterVisitor {
    rec: IterRecord,
}

impl Visit for IterVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        let v = value as Number;
        match field.name() {
            "objective" => self.rec.objective = v,
            "inf_pr" => self.rec.inf_pr = v,
            "inf_du" => self.rec.inf_du = v,
            "mu" => self.rec.mu = v,
            "d_norm" => self.rec.d_norm = v,
            "regularization" => self.rec.regularization = v,
            "alpha_dual" => self.rec.alpha_dual = v,
            "alpha_primal" => self.rec.alpha_primal = v,
            _ => {}
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        match field.name() {
            "iter" => self.rec.iter = value as Index,
            "ls_trials" => self.rec.ls_trials = value as Index,
            _ => {}
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        // Match the field directly rather than routing through
        // `record_i64` (which would `value as i64`-truncate a value
        // above `i64::MAX`). `iter`/`ls_trials` never approach that, but
        // matching here keeps the cast localized to the two real fields.
        match field.name() {
            "iter" => self.rec.iter = value as Index,
            "ls_trials" => self.rec.ls_trials = value as Index,
            _ => {}
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "alpha_char" {
            self.rec.alpha_primal_char = value.chars().next().unwrap_or(' ');
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {
        // The message and any Debug-formatted fields are irrelevant to
        // the numeric record.
    }
}

// ---- Collector layer ----

/// `tracing` layer that rebuilds [`IterRecord`]s from [`ITER_TARGET`]
/// events into the active [`IterCaptureGuard`] slot.
#[derive(Debug, Default, Clone)]
pub struct IterCollectorLayer;

impl<S> Layer<S> for IterCollectorLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        if event.metadata().target() != ITER_TARGET {
            return;
        }
        // Skip iterations belonging to a restoration sub-solve so the
        // report keeps only the outer trajectory.
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                if span.name() == RESTORATION_SPAN {
                    return;
                }
            }
        }
        let mut visitor = IterVisitor::default();
        event.record(&mut visitor);
        push_record(visitor.rec);
    }
}

/// Per-layer filter for [`IterCollectorLayer`]: admit spans (so the
/// collector's `event_scope` can see the `restoration` ancestor for
/// scoping) plus the iteration event itself.
///
/// A per-layer filter is required so the collector does not force every
/// callsite globally enabled; without admitting spans here the filtered
/// `Context` would hide span ancestry. Defined once and reused at every
/// `with_filter` site — note the `with_filter` *call* must still live in
/// each branch, because the resulting `Filtered` type carries the
/// per-branch subscriber type parameter.
fn collector_admits(m: &tracing::Metadata<'_>) -> bool {
    m.is_span() || m.target() == ITER_TARGET
}

// ---- Tiger/rust themed text formatter ----

/// Foreground style for a log level in the tiger/rust theme.
fn level_style(level: tracing::Level) -> anstyle::Style {
    use pounce_common::style::{ALPHA_HOT, TAN, TIGER_ORANGE};
    let color = match level {
        tracing::Level::ERROR => ALPHA_HOT,
        tracing::Level::WARN => TIGER_ORANGE,
        tracing::Level::INFO => TAN,
        tracing::Level::DEBUG => anstyle::RgbColor(0x9a, 0x8c, 0x70),
        tracing::Level::TRACE => anstyle::RgbColor(0x6a, 0x5d, 0x48),
    };
    anstyle::Style::new().fg_color(Some(anstyle::Color::Rgb(color)))
}

/// Compact event formatter: `LEVEL target: message field=…`, with the
/// level rendered in the tiger/rust palette when ANSI is enabled.
struct TigerFormat;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for TigerFormat
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let meta = event.metadata();
        let level = *meta.level();
        if writer.has_ansi_escapes() {
            let style = level_style(level);
            write!(
                writer,
                "{}{:>5}{} ",
                style.render(),
                level,
                style.render_reset()
            )?;
        } else {
            write!(writer, "{level:>5} ")?;
        }
        write!(writer, "{}: ", meta.target())?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

// ---- Subscriber install ----

/// Install the global tracing subscriber for a normal run. Idempotent
/// (`try_init`): safe to call from multiple frontends or repeated
/// Python imports.
///
/// Reads `RUST_LOG` (filtering, default `info`), `POUNCE_LOG_FORMAT`
/// (`text` | `json`), and `NO_COLOR`/`CLICOLOR_FORCE` (color policy).
pub fn init_subscriber() {
    install();
}

/// Install a subscriber suitable for tests: same layers as
/// [`init_subscriber`], so iteration capture works under
/// [`IterCaptureGuard`]. Idempotent.
///
/// Currently identical to [`init_subscriber`]; kept as a distinct entry
/// point so test setup can diverge (e.g. an in-memory sink) without
/// touching the production install path. Tests needing an *isolated*
/// subscriber should build their own with `with_default` instead.
pub fn init_for_tests() {
    install();
}

fn install() {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::filter::filter_fn;
    use tracing_subscriber::prelude::*;

    // Bridge the `log` crate into `tracing` so any remaining `log::*`
    // call sites — chiefly transitive dependencies — surface through
    // our subscriber and obey `RUST_LOG`. Idempotent; the `Err` when a
    // logger is already installed is intentionally ignored.
    let _ = tracing_log::LogTracer::init();

    let want_json = std::env::var("POUNCE_LOG_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    // Record the JSON-sink decision so `iteration_event_wanted()` keeps
    // emitting the per-iteration event for the stderr stream even when
    // no in-process capture is active.
    JSON_LOGGING.store(want_json, std::sync::atomic::Ordering::Relaxed);

    // The collector only ever wants the iteration event; it must NOT be
    // subject to the console's `pounce::iteration=off` suppression, so
    // it carries its own target filter. It is constructed inside each
    // branch so its subscriber type parameter is inferred per-branch.
    if want_json {
        let collector = IterCollectorLayer.with_filter(filter_fn(collector_admits));
        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_writer(std::io::stderr)
            .with_filter(env_filter());
        let _ = tracing_subscriber::registry()
            .with(json_layer)
            .with(collector)
            .try_init();
    } else {
        let collector = IterCollectorLayer.with_filter(filter_fn(collector_admits));
        let ansi = ansi_enabled();
        let text_layer = tracing_subscriber::fmt::layer()
            .event_format(TigerFormat)
            .with_ansi(ansi)
            .with_writer(std::io::stderr)
            .with_filter(console_filter());
        let _ = tracing_subscriber::registry()
            .with(text_layer)
            .with(collector)
            .try_init();
    }

    /// `RUST_LOG` filter, defaulting to `info`.
    fn env_filter() -> EnvFilter {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    }

    /// Console filter: `RUST_LOG` plus suppression of the iteration
    /// target (its human form is the colored table on stdout).
    fn console_filter() -> EnvFilter {
        let base = env_filter();
        match format!("{ITER_TARGET}=off").parse() {
            Ok(directive) => base.add_directive(directive),
            Err(_) => base,
        }
    }

    /// ANSI on unless `NO_COLOR`, with `CLICOLOR_FORCE` overriding and
    /// a terminal-capability check otherwise.
    fn ansi_enabled() -> bool {
        if anstyle_query::clicolor_force() {
            return true;
        }
        if anstyle_query::no_color() {
            return false;
        }
        anstyle_query::term_supports_ansi_color()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(iter: i32, alpha: f64, c: char) -> IterRecord {
        IterRecord {
            iter,
            objective: 1.0,
            inf_pr: 2.0,
            inf_du: 3.0,
            mu: 4.0,
            d_norm: 5.0,
            regularization: 6.0,
            alpha_dual: 7.0,
            alpha_primal: alpha,
            alpha_primal_char: c,
            ls_trials: 1,
        }
    }

    #[test]
    fn iteration_event_wanted_tracks_active_capture() {
        // Default (no capture on this thread, JSON sink off): nothing
        // consumes the event, so it should be suppressed.
        assert!(!iteration_event_wanted());
        let guard = IterCaptureGuard::start();
        assert!(iteration_event_wanted(), "capture active → event wanted");
        let _ = guard.finish();
        assert!(
            !iteration_event_wanted(),
            "capture ended → event suppressed"
        );
    }

    #[test]
    fn guard_captures_pushed_records() {
        let guard = IterCaptureGuard::start();
        push_record(sample_record(0, 1.0, ' '));
        push_record(sample_record(1, 0.5, 'R'));
        let got = guard.finish();
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].iter, 1);
        assert_eq!(got[1].alpha_primal_char, 'R');
    }

    #[test]
    fn no_guard_means_records_are_dropped() {
        // Without an active guard, push is a no-op (no panic, no leak).
        push_record(sample_record(0, 1.0, ' '));
        let guard = IterCaptureGuard::start();
        let got = guard.finish();
        assert!(got.is_empty());
    }

    #[test]
    fn guard_restores_previous_slot_on_finish() {
        let outer = IterCaptureGuard::start();
        push_record(sample_record(0, 1.0, ' '));
        {
            let inner = IterCaptureGuard::start();
            push_record(sample_record(99, 0.1, 'R'));
            let inner_got = inner.finish();
            assert_eq!(inner_got.len(), 1);
            assert_eq!(inner_got[0].iter, 99);
        }
        // Outer slot must still hold only its own record.
        push_record(sample_record(1, 1.0, ' '));
        let outer_got = outer.finish();
        assert_eq!(outer_got.len(), 2);
        assert_eq!(outer_got[0].iter, 0);
        assert_eq!(outer_got[1].iter, 1);
    }

    #[test]
    fn collector_excludes_restoration_nested_iterations() {
        use tracing_subscriber::filter::filter_fn;
        use tracing_subscriber::prelude::*;

        fn emit(iter: i64, ch: char) {
            let s = ch.to_string();
            tracing::info!(
                target: ITER_TARGET,
                iter = iter,
                objective = 0.0,
                alpha_primal = 1.0,
                alpha_char = s.as_str(),
            );
        }

        // Same layer wiring as `install()`: the filter must admit spans
        // so the collector's `event_scope` can see the `restoration`
        // ancestor. Regression guard for the per-layer-filter bug where
        // a `target`-only filter hid span ancestry and let inner
        // restoration iterations leak into the report.
        let collector = IterCollectorLayer.with_filter(filter_fn(collector_admits));
        let subscriber = tracing_subscriber::registry().with(collector);

        let captured = tracing::subscriber::with_default(subscriber, || {
            let guard = IterCaptureGuard::start();
            emit(0, ' '); // outer -> captured
            {
                let _resto = tracing::info_span!("restoration").entered();
                let _inner_solve = tracing::info_span!("solve").entered();
                let _inner_iter = tracing::info_span!("iteration").entered();
                emit(99, 'R'); // inner restoration sub-solve -> excluded
            }
            emit(1, ' '); // outer -> captured
            guard.finish()
        });

        let iters: Vec<i32> = captured.iter().map(|r| r.iter).collect();
        assert_eq!(
            iters,
            vec![0, 1],
            "inner restoration iteration leaked: {iters:?}"
        );
    }

    #[test]
    fn log_records_bridge_into_tracing() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::prelude::*;

        // Minimal layer that records each event's `message` field.
        #[derive(Clone)]
        struct CaptureLayer {
            buf: Arc<Mutex<Vec<String>>>,
        }
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLayer {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                struct V<'a>(&'a mut Vec<String>);
                impl tracing::field::Visit for V<'_> {
                    fn record_debug(&mut self, f: &Field, value: &dyn std::fmt::Debug) {
                        if f.name() == "message" {
                            self.0.push(format!("{value:?}"));
                        }
                    }
                }
                let mut g = self.buf.lock().unwrap_or_else(|p| p.into_inner());
                event.record(&mut V(&mut g));
            }
        }

        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(CaptureLayer { buf: buf.clone() });

        // The same bridge `install()` sets up. Global + idempotent.
        let _ = tracing_log::LogTracer::init();
        tracing::subscriber::with_default(subscriber, || {
            // A `log` record as a transitive dependency would emit.
            log::error!(target: "some_transitive_dep", "bridged log record");
        });

        let got = buf.lock().unwrap_or_else(|p| p.into_inner());
        assert!(
            got.iter().any(|m| m.contains("bridged log record")),
            "log record did not reach the tracing layer; captured: {got:?}"
        );
    }

    #[test]
    fn iter_record_default_and_assignment() {
        // This checks `IterRecord` field assignment, not the `Visit`
        // impl — constructing a real `tracing::field::Field` standalone
        // needs a callsite, so the visitor's record_* arms are covered
        // end-to-end by `collector_excludes_restoration_nested_iterations`
        // (which emits real events and asserts the rebuilt `iter`s).
        let mut v = IterVisitor::default();
        v.rec.iter = 7;
        v.rec.alpha_primal = 0.25;
        v.rec.alpha_primal_char = 'S';
        assert_eq!(v.rec.iter, 7);
        assert_eq!(v.rec.alpha_primal_char, 'S');
    }
}
