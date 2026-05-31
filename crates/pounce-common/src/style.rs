//! Tiger / rust / warm branded color palette for POUNCE output
//! (pounce#71).
//!
//! This module is **pure** — every function maps values to
//! [`anstyle`] colors/styles with no I/O and no global state, so the
//! whole palette is unit-testable without a TTY. Terminal-capability
//! detection ([`truecolor_enabled`], [`color_enabled_stdout`]) reads
//! the environment but never emits anything; the actual color
//! *stripping* for the iteration table is delegated to
//! `anstream::AutoStream` at the print site.
//!
//! Two orthogonal channels drive the iteration table:
//!
//! * **Background** marks *restoration* lines, keyed off the
//!   per-iteration `alpha_primal_char` tag — `'s'` soft-stay → tan,
//!   `'S'` soft-exit → amber, `'R'` (and the dedicated restoration
//!   phase's `r`-suffixed rows) → deep rust.
//! * **Foreground** is a smooth gradient driven by the primal step
//!   length `alpha ∈ [0, 1]`. On normal lines it runs black (α = 1,
//!   full Newton step) → red (α → 0, stalling). On restoration lines
//!   the ramp shifts to cream → bright-yellow so the text stays
//!   legible on the dark background.

use anstyle::{Ansi256Color, Color, RgbColor, Style};

// ---- Palette constants (tiger / rust / warm) ----

/// Background for hard restoration (`'R'` + dedicated resto-phase rows).
pub const RUST_DEEP: RgbColor = RgbColor(0x6e, 0x26, 0x0e);
/// Background for soft-restoration "exit" (`'S'`).
pub const AMBER: RgbColor = RgbColor(0xb5, 0x6a, 0x12);
/// Background for soft-restoration "stay" (`'s'`).
pub const TAN: RgbColor = RgbColor(0x8a, 0x6d, 0x3b);
/// Accent used for `WARN`-level logs and banners.
pub const TIGER_ORANGE: RgbColor = RgbColor(0xe8, 0x7a, 0x1e);
/// Foreground on restoration lines at α = 1 (full step).
pub const CREAM: RgbColor = RgbColor(0xf5, 0xe6, 0xc8);
/// Foreground on restoration lines at α → 0 (stalling).
pub const BRIGHT_YEL: RgbColor = RgbColor(0xff, 0xe0, 0x3a);
/// Foreground on normal lines at α → 0 (stalling) — the "hot" red.
pub const ALPHA_HOT: RgbColor = RgbColor(0xcc, 0x22, 0x00);
/// Foreground on normal lines at α = 1 (full Newton step).
pub const ALPHA_COOL: RgbColor = RgbColor(0x00, 0x00, 0x00);

// ---- Restoration kind ↔ color ----

/// `true` when `c` denotes a restoration line (`'s'`, `'S'`, `'R'`).
/// `'t'`/`'T'` (tiny step) are deliberately excluded — that stalling
/// condition is conveyed by the foreground gradient, not a background.
pub fn is_resto_char(c: char) -> bool {
    matches!(c, 's' | 'S' | 'R')
}

/// Map the iteration's `alpha_primal_char` to its restoration
/// background, or `None` for a normal (non-restoration) line.
pub fn resto_background_rgb(c: char) -> Option<RgbColor> {
    match c {
        's' => Some(TAN),
        'S' => Some(AMBER),
        'R' => Some(RUST_DEEP),
        _ => None,
    }
}

/// Human-readable restoration kind for the structured
/// `pounce::iteration` tracing event.
pub fn resto_kind_str(c: char) -> &'static str {
    match c {
        's' => "soft_stay",
        'S' => "soft_exit",
        'R' => "hard",
        _ => "none",
    }
}

// ---- Alpha gradient ----

/// Linear interpolation of one 8-bit channel by `t ∈ [0, 1]`.
fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    let v = a as f64 + (b as f64 - a as f64) * t;
    v.round().clamp(0.0, 255.0) as u8
}

/// The raw RGB foreground for primal step length `alpha`. `in_resto`
/// selects the cream→bright-yellow ramp instead of the black→red one.
///
/// `alpha` is clamped to `[0, 1]`; non-finite input is treated as a
/// full step (`alpha = 1`). The interpolation parameter is `1 - alpha`
/// so that α = 1 is "cool" (black / cream) and α → 0 is "hot"
/// (red / bright-yellow).
pub fn alpha_gradient_rgb(alpha: f64, in_resto: bool) -> RgbColor {
    let alpha = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let t = 1.0 - alpha;
    let (cool, hot) = if in_resto {
        (CREAM, BRIGHT_YEL)
    } else {
        (ALPHA_COOL, ALPHA_HOT)
    };
    RgbColor(
        lerp_u8(cool.0, hot.0, t),
        lerp_u8(cool.1, hot.1, t),
        lerp_u8(cool.2, hot.2, t),
    )
}

// ---- Truecolor → 256-color downgrade ----

/// Snap an 8-bit value to its nearest index on the xterm 6×6×6 color
/// cube's per-channel step ladder `[0, 95, 135, 175, 215, 255]`.
fn cube_level(v: u8) -> u8 {
    const STEPS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let mut best = 0u8;
    let mut best_d = u16::MAX;
    for (i, &s) in STEPS.iter().enumerate() {
        let d = (v as i16 - s as i16).unsigned_abs();
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

/// Nearest xterm-256 cube color to an RGB triple. Used as the graceful
/// fallback on terminals that advertise ANSI color but not truecolor.
pub fn nearest_ansi256(c: RgbColor) -> Ansi256Color {
    let r = cube_level(c.0);
    let g = cube_level(c.1);
    let b = cube_level(c.2);
    Ansi256Color(16 + 36 * r + 6 * g + b)
}

/// Wrap an RGB color as an [`anstyle::Color`], downgrading to the
/// nearest 256-color when the terminal lacks truecolor support.
pub fn downgrade(c: RgbColor, truecolor: bool) -> Color {
    if truecolor {
        Color::Rgb(c)
    } else {
        Color::Ansi256(nearest_ansi256(c))
    }
}

// ---- Composed iteration-row style ----

/// Build the [`Style`] for one iteration-table row: foreground from the
/// alpha gradient, optional background from the restoration kind.
/// Honors the detected truecolor capability so the same call yields
/// RGB on capable terminals and a 256-color approximation elsewhere.
pub fn iteration_row_style(alpha_primal: f64, alpha_char: char) -> Style {
    iteration_row_style_with(alpha_primal, alpha_char, truecolor_enabled())
}

/// [`iteration_row_style`] with the truecolor decision injected — the
/// unit-test seam (no environment reads).
pub fn iteration_row_style_with(alpha_primal: f64, alpha_char: char, truecolor: bool) -> Style {
    let in_resto = is_resto_char(alpha_char);
    let fg = downgrade(alpha_gradient_rgb(alpha_primal, in_resto), truecolor);
    let mut style = Style::new().fg_color(Some(fg));
    if let Some(bg) = resto_background_rgb(alpha_char) {
        style = style.bg_color(Some(downgrade(bg, truecolor)));
    }
    style
}

// ---- Terminal-capability detection ----

/// `true` when the terminal advertises 24-bit truecolor (`COLORTERM`).
pub fn truecolor_enabled() -> bool {
    anstyle_query::truecolor()
}

/// `true` when colored output should be emitted to stdout: stdout is a
/// terminal and the user has not opted out via `NO_COLOR` (unless
/// `CLICOLOR_FORCE` overrides). Stream-based call sites should prefer
/// `anstream::AutoStream`, which applies the same policy while
/// stripping escapes from redirected output; this helper is for code
/// that must branch without a stream handle.
pub fn color_enabled_stdout() -> bool {
    use std::io::IsTerminal;
    if anstyle_query::clicolor_force() {
        return true;
    }
    if anstyle_query::no_color() {
        return false;
    }
    std::io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resto_background_maps_three_kinds() {
        assert_eq!(resto_background_rgb('s'), Some(TAN));
        assert_eq!(resto_background_rgb('S'), Some(AMBER));
        assert_eq!(resto_background_rgb('R'), Some(RUST_DEEP));
        // Non-restoration tags (including tiny-step) get no background.
        for c in [' ', 'f', 'h', 'w', 'W', 't', 'T'] {
            assert_eq!(resto_background_rgb(c), None, "char {c:?}");
        }
    }

    #[test]
    fn is_resto_char_only_for_s_caps_r() {
        for c in ['s', 'S', 'R'] {
            assert!(is_resto_char(c), "char {c:?}");
        }
        for c in [' ', 'f', 'h', 'w', 'W', 't', 'T'] {
            assert!(!is_resto_char(c), "char {c:?}");
        }
    }

    #[test]
    fn alpha_gradient_normal_endpoints() {
        // Full step → black; stalled → hot red.
        assert_eq!(alpha_gradient_rgb(1.0, false), ALPHA_COOL);
        assert_eq!(alpha_gradient_rgb(0.0, false), ALPHA_HOT);
    }

    #[test]
    fn alpha_gradient_resto_endpoints() {
        assert_eq!(alpha_gradient_rgb(1.0, true), CREAM);
        assert_eq!(alpha_gradient_rgb(0.0, true), BRIGHT_YEL);
    }

    #[test]
    fn alpha_gradient_is_monotonic_toward_hot() {
        // As alpha shrinks the red channel must not decrease (normal
        // ramp drives from black 0x00 → 0xcc).
        let mut prev = alpha_gradient_rgb(1.0, false).0;
        for step in 1..=10 {
            let a = 1.0 - step as f64 / 10.0;
            let r = alpha_gradient_rgb(a, false).0;
            assert!(r >= prev, "alpha={a} red went backwards {prev}->{r}");
            prev = r;
        }
        assert_eq!(prev, ALPHA_HOT.0);
    }

    #[test]
    fn alpha_gradient_clamps_and_handles_nonfinite() {
        assert_eq!(alpha_gradient_rgb(2.0, false), ALPHA_COOL);
        assert_eq!(alpha_gradient_rgb(-1.0, false), ALPHA_HOT);
        // NaN is treated as a full step (no false stalling alarm).
        assert_eq!(alpha_gradient_rgb(f64::NAN, false), ALPHA_COOL);
    }

    #[test]
    fn downgrade_picks_rgb_or_256() {
        assert_eq!(downgrade(RUST_DEEP, true), Color::Rgb(RUST_DEEP));
        // 256-color path yields a cube index, never an RGB color.
        match downgrade(RUST_DEEP, false) {
            Color::Ansi256(_) => {}
            other => panic!("expected Ansi256, got {other:?}"),
        }
    }

    #[test]
    fn nearest_ansi256_snaps_pure_colors() {
        // Pure white → cube corner 231 (16 + 36*5 + 6*5 + 5).
        assert_eq!(
            nearest_ansi256(RgbColor(0xff, 0xff, 0xff)),
            Ansi256Color(231)
        );
        // Pure black → cube origin 16.
        assert_eq!(
            nearest_ansi256(RgbColor(0x00, 0x00, 0x00)),
            Ansi256Color(16)
        );
    }

    #[test]
    fn iteration_row_style_composes_fg_and_bg() {
        // Restoration row: both fg gradient and bg present.
        let s = iteration_row_style_with(0.5, 'R', true);
        assert!(s.get_fg_color().is_some());
        assert_eq!(s.get_bg_color(), Some(Color::Rgb(RUST_DEEP)));
        // Normal row: fg only, no background.
        let n = iteration_row_style_with(1.0, ' ', true);
        assert_eq!(n.get_fg_color(), Some(Color::Rgb(ALPHA_COOL)));
        assert_eq!(n.get_bg_color(), None);
    }
}
