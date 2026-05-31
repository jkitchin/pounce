//! Integration test for the colored iteration table (pounce#71).
//!
//! The table is styled at the print site by wrapping the *plain*
//! `format_row` string in an `anstyle::Style`. These tests assert the
//! two invariants that keep that safe:
//!
//! 1. Styling adds only zero-width SGR escapes — stripping them must
//!    return the original plain row byte-for-byte (column alignment is
//!    preserved).
//! 2. When color is stripped (redirected output / `NO_COLOR`), the
//!    bytes are exactly the plain row with no escapes.

use pounce_common::style::iteration_row_style_with;

/// A representative plain iteration row (fixed-width, as produced by
/// `OrigIterationOutput::format_row`).
const PLAIN_ROW: &str =
    "   7  1.2340000e+00 1.00e-02 2.00e-03   -3.0 4.00e-01      - 5.00e-01 6.00e-01R  2";

fn style_row(plain: &str, alpha: f64, alpha_char: char, truecolor: bool) -> String {
    let style = iteration_row_style_with(alpha, alpha_char, truecolor);
    format!("{}{}{}", style.render(), plain, style.render_reset())
}

#[test]
fn styled_row_strips_back_to_plain() {
    // Restoration row with a mid-range alpha — exercises both fg
    // gradient and bg.
    let styled = style_row(PLAIN_ROW, 0.4, 'R', true);
    assert!(
        styled.contains('\u{1b}'),
        "expected ANSI escapes: {styled:?}"
    );
    let stripped = anstream::adapter::strip_str(&styled).to_string();
    assert_eq!(stripped, PLAIN_ROW, "stripping must recover the plain row");
}

#[test]
fn normal_row_strips_back_to_plain() {
    let styled = style_row(PLAIN_ROW, 1.0, ' ', true);
    let stripped = anstream::adapter::strip_str(&styled).to_string();
    assert_eq!(stripped, PLAIN_ROW);
}

#[test]
fn anstream_strips_when_not_a_terminal() {
    // Writing styled bytes through a StripStream (what an `AutoStream`
    // degrades to for a non-TTY sink) must yield plain text.
    use std::io::Write as _;
    let styled = style_row(PLAIN_ROW, 0.1, 'S', true);
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut s = anstream::StripStream::new(&mut buf);
        s.write_all(styled.as_bytes()).unwrap();
        s.flush().unwrap();
    }
    assert_eq!(String::from_utf8(buf).unwrap(), PLAIN_ROW);
}

#[test]
fn non_truecolor_uses_256_but_still_strips_clean() {
    // The 256-color downgrade path must also be zero-width.
    let styled = style_row(PLAIN_ROW, 0.5, 'R', false);
    assert!(styled.contains('\u{1b}'));
    let stripped = anstream::adapter::strip_str(&styled).to_string();
    assert_eq!(stripped, PLAIN_ROW);
}
