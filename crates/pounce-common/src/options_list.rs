//! User-set options list.
//!
//! Mirrors `Common/IpOptionsList.{hpp,cpp}`. Stores name → value
//! string mappings; lookup is case-insensitive and prefix-aware. The
//! prefix mechanism is what gives the restoration sub-algorithm its
//! own option scope: looking up `tol` with prefix `"resto."` first
//! tries `resto.tol`, then falls back to `tol`.
//!
//! Internal value representation is always a `String`, exactly as in
//! upstream — typed accessors parse on each call, matching Ipopt
//! behavior.

use crate::exception::{ExceptionKind, SolverException};
use crate::reg_options::{DefaultValue, OptionType, RegisteredOptions};
use crate::throw;
use crate::types::{Index, Number};
use std::collections::BTreeMap;
use std::io::Read;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct OptionValue {
    value: String,
    counter: std::cell::Cell<Index>,
    allow_clobber: bool,
    dont_print: bool,
}

impl OptionValue {
    fn new(value: String, allow_clobber: bool, dont_print: bool) -> Self {
        Self {
            value,
            counter: std::cell::Cell::new(0),
            allow_clobber,
            dont_print,
        }
    }
    fn get_value(&self) -> &str {
        self.counter.set(self.counter.get() + 1);
        &self.value
    }
}

/// Mirrors `Ipopt::OptionsList`.
#[derive(Debug, Default, Clone)]
pub struct OptionsList {
    options: BTreeMap<String, OptionValue>,
    reg_options: Option<Rc<RegisteredOptions>>,
}

impl OptionsList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_registered(reg: Rc<RegisteredOptions>) -> Self {
        Self {
            options: BTreeMap::new(),
            reg_options: Some(reg),
        }
    }

    pub fn set_registered_options(&mut self, reg: Rc<RegisteredOptions>) {
        self.reg_options = Some(reg);
    }

    pub fn registered_options(&self) -> Option<Rc<RegisteredOptions>> {
        self.reg_options.clone()
    }

    pub fn clear(&mut self) {
        self.options.clear();
    }

    /// Every option name present in the list, lowercased, in sorted
    /// order.
    ///
    /// "Present" means someone called a `set_*_value` for it — an unset
    /// option is absent here even though it *reads back* as its
    /// registered default. That distinction is the whole point of the
    /// accessor: it answers "what did this run actually mention?",
    /// which is not a question `get_*_value` can answer.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.options.keys().map(String::as_str)
    }

    fn key(name: &str) -> String {
        name.to_ascii_lowercase()
    }

    /// Mirrors `OptionsList::find_tag` — try `prefix+tag` first, then
    /// bare `tag`. Returns the stored string and bumps its read counter.
    fn find_tag(&self, tag: &str, prefix: &str) -> Option<&OptionValue> {
        if !prefix.is_empty() {
            let key = Self::key(&format!("{prefix}{tag}"));
            if let Some(v) = self.options.get(&key) {
                return Some(v);
            }
        }
        self.options.get(&Self::key(tag))
    }

    fn will_allow_clobber(&self, tag: &str) -> bool {
        match self.options.get(&Self::key(tag)) {
            Some(v) => v.allow_clobber,
            None => true,
        }
    }

    /// Mirrors `SetStringValue`.
    pub fn set_string_value(
        &mut self,
        tag: &str,
        value: &str,
        allow_clobber: bool,
        dont_print: bool,
    ) -> Result<bool, SolverException> {
        if let Some(reg) = &self.reg_options {
            let opt = reg.get_option(tag).ok_or_else(|| {
                SolverException::new(
                    ExceptionKind::OPTION_INVALID,
                    format!("Unknown option \"{tag}\"."),
                    file!(),
                    line!() as Index,
                )
            })?;
            if opt.option_type != OptionType::OT_String {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\" is not a string option.")
                );
            }
            if !opt.is_valid_string(value) {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Invalid value \"{value}\" for string option \"{tag}\".")
                );
            }
        }
        if !self.will_allow_clobber(tag) {
            return Ok(false);
        }
        let stored = value.to_ascii_lowercase();
        self.options.insert(
            Self::key(tag),
            OptionValue::new(stored, allow_clobber, dont_print),
        );
        Ok(true)
    }

    /// Mirrors `SetNumericValue`.
    pub fn set_numeric_value(
        &mut self,
        tag: &str,
        value: Number,
        allow_clobber: bool,
        dont_print: bool,
    ) -> Result<bool, SolverException> {
        if let Some(reg) = &self.reg_options {
            let opt = reg.get_option(tag).ok_or_else(|| {
                SolverException::new(
                    ExceptionKind::OPTION_INVALID,
                    format!("Unknown option \"{tag}\"."),
                    file!(),
                    line!() as Index,
                )
            })?;
            if opt.option_type != OptionType::OT_Number {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\" is not a numeric option.")
                );
            }
            if !opt.is_valid_number(value) {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Numeric value {value} for option \"{tag}\" out of range.")
                );
            }
        }
        if !self.will_allow_clobber(tag) {
            return Ok(false);
        }
        // Print with full precision so round-trip preserves the value.
        let s = format!("{value:.18e}");
        self.options.insert(
            Self::key(tag),
            OptionValue::new(s, allow_clobber, dont_print),
        );
        Ok(true)
    }

    /// Mirrors `SetIntegerValue`.
    pub fn set_integer_value(
        &mut self,
        tag: &str,
        value: Index,
        allow_clobber: bool,
        dont_print: bool,
    ) -> Result<bool, SolverException> {
        if let Some(reg) = &self.reg_options {
            let opt = reg.get_option(tag).ok_or_else(|| {
                SolverException::new(
                    ExceptionKind::OPTION_INVALID,
                    format!("Unknown option \"{tag}\"."),
                    file!(),
                    line!() as Index,
                )
            })?;
            if opt.option_type != OptionType::OT_Integer {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\" is not an integer option.")
                );
            }
            if !opt.is_valid_integer(value) {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Integer value {value} for option \"{tag}\" out of range.")
                );
            }
        }
        if !self.will_allow_clobber(tag) {
            return Ok(false);
        }
        self.options.insert(
            Self::key(tag),
            OptionValue::new(value.to_string(), allow_clobber, dont_print),
        );
        Ok(true)
    }

    /// Mirrors `SetBoolValue`.
    pub fn set_bool_value(
        &mut self,
        tag: &str,
        value: bool,
        allow_clobber: bool,
        dont_print: bool,
    ) -> Result<bool, SolverException> {
        self.set_string_value(
            tag,
            if value { "yes" } else { "no" },
            allow_clobber,
            dont_print,
        )
    }

    /// Mirrors `UnsetValue`. Returns true if the value was removed.
    pub fn unset_value(&mut self, tag: &str) -> bool {
        let key = Self::key(tag);
        if let Some(v) = self.options.get(&key) {
            if !v.allow_clobber {
                return false;
            }
            self.options.remove(&key);
            true
        } else {
            false
        }
    }

    /// Mirrors `GetStringValue`. Returns true if found in the list.
    /// Falls back to the registered default when not found.
    pub fn get_string_value(
        &self,
        tag: &str,
        prefix: &str,
    ) -> Result<(String, bool), SolverException> {
        if let Some(v) = self.find_tag(tag, prefix) {
            return Ok((v.get_value().to_string(), true));
        }
        if let Some(reg) = &self.reg_options {
            if let Some(opt) = reg.get_option(tag) {
                if let DefaultValue::String(d) = &opt.default {
                    return Ok((d.clone(), false));
                }
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\" is not a string option.")
                );
            }
        }
        Ok((String::new(), false))
    }

    /// Mirrors `GetNumericValue`.
    pub fn get_numeric_value(
        &self,
        tag: &str,
        prefix: &str,
    ) -> Result<(Number, bool), SolverException> {
        if let Some(v) = self.find_tag(tag, prefix) {
            let s = v.get_value().to_string();
            let parsed = parse_ipopt_number(&s).ok_or_else(|| {
                SolverException::new(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\": cannot parse value \"{s}\" as Number."),
                    file!(),
                    line!() as Index,
                )
            })?;
            return Ok((parsed, true));
        }
        if let Some(reg) = &self.reg_options {
            if let Some(opt) = reg.get_option(tag) {
                if let DefaultValue::Number(d) = &opt.default {
                    return Ok((*d, false));
                }
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\" is not a numeric option.")
                );
            }
        }
        Ok((0.0, false))
    }

    /// Mirrors `GetIntegerValue`.
    pub fn get_integer_value(
        &self,
        tag: &str,
        prefix: &str,
    ) -> Result<(Index, bool), SolverException> {
        if let Some(v) = self.find_tag(tag, prefix) {
            let s = v.get_value().to_string();
            let parsed: Index = s.trim().parse().map_err(|_| {
                SolverException::new(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\": cannot parse value \"{s}\" as Integer."),
                    file!(),
                    line!() as Index,
                )
            })?;
            return Ok((parsed, true));
        }
        if let Some(reg) = &self.reg_options {
            if let Some(opt) = reg.get_option(tag) {
                if let DefaultValue::Integer(d) = &opt.default {
                    return Ok((*d, false));
                }
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Option \"{tag}\" is not an integer option.")
                );
            }
        }
        Ok((0, false))
    }

    /// Mirrors `GetBoolValue`. Accepts `"yes"`/`"no"`.
    pub fn get_bool_value(&self, tag: &str, prefix: &str) -> Result<(bool, bool), SolverException> {
        let (s, found) = self.get_string_value(tag, prefix)?;
        let v = match s.to_ascii_lowercase().as_str() {
            "yes" => true,
            "no" => false,
            other => throw!(
                ExceptionKind::OPTION_INVALID,
                format!("Option \"{tag}\" has non-boolean value \"{other}\".")
            ),
        };
        Ok((v, found))
    }

    /// Mirrors `GetEnumValue`. Returns the index of the value in the
    /// registered string list.
    pub fn get_enum_value(
        &self,
        tag: &str,
        prefix: &str,
    ) -> Result<(Index, bool), SolverException> {
        let (s, found) = self.get_string_value(tag, prefix)?;
        let reg = self.reg_options.as_ref().ok_or_else(|| {
            SolverException::new(
                ExceptionKind::OPTION_INVALID,
                "GetEnumValue requires a RegisteredOptions registry.".to_string(),
                file!(),
                line!() as Index,
            )
        })?;
        let opt = reg.get_option(tag).ok_or_else(|| {
            SolverException::new(
                ExceptionKind::OPTION_INVALID,
                format!("Unknown option \"{tag}\"."),
                file!(),
                line!() as Index,
            )
        })?;
        let idx = opt.map_string_to_enum(&s).ok_or_else(|| {
            SolverException::new(
                ExceptionKind::ERROR_CONVERTING_STRING_TO_ENUM,
                format!("Cannot map \"{s}\" to enum for option \"{tag}\"."),
                file!(),
                line!() as Index,
            )
        })?;
        Ok((idx, found))
    }

    /// Mirrors `ReadFromStream`. Parses an `ipopt.opt`-style file:
    /// whitespace-separated `tag value` pairs, `#` line comments,
    /// double-quoted tokens permitted.
    pub fn read_from_stream<R: Read>(
        &mut self,
        mut r: R,
        allow_clobber: bool,
    ) -> Result<(), SolverException> {
        let mut s = String::new();
        r.read_to_string(&mut s).map_err(|e| {
            SolverException::new(
                ExceptionKind::OPTION_INVALID,
                format!("I/O error reading options: {e}"),
                file!(),
                line!() as Index,
            )
        })?;
        self.read_from_str(&s, allow_clobber)
    }

    pub fn read_from_str(&mut self, s: &str, allow_clobber: bool) -> Result<(), SolverException> {
        let mut tokens = Tokenizer::new(s);
        loop {
            let Some(tag) = tokens.next_token()? else {
                return Ok(());
            };
            let Some(value) = tokens.next_token()? else {
                throw!(
                    ExceptionKind::OPTION_INVALID,
                    format!("Error reading value for tag {tag} from option file.")
                );
            };
            self.set_from_text(&tag, &value, allow_clobber)?;
        }
    }

    fn set_from_text(
        &mut self,
        tag: &str,
        value: &str,
        allow_clobber: bool,
    ) -> Result<(), SolverException> {
        if let Some(reg) = self.reg_options.clone() {
            let opt = reg.get_option(tag).ok_or_else(|| SolverException::new(
                ExceptionKind::OPTION_INVALID,
                format!("Read Option: \"{tag}\". It is not a valid option. Check the list of available options."),
                file!(), line!() as Index,
            ))?;
            match opt.option_type {
                OptionType::OT_String => {
                    let ok = self.set_string_value(tag, value, allow_clobber, false)?;
                    if !ok {
                        throw!(
                            ExceptionKind::OPTION_INVALID,
                            "Error setting string value read from option file.".to_string()
                        );
                    }
                }
                OptionType::OT_Number => {
                    let v = parse_ipopt_number(value).ok_or_else(|| SolverException::new(
                        ExceptionKind::OPTION_INVALID,
                        format!("Option \"{tag}\": Double value expected, but non-numeric option value \"{value}\" found.\n"),
                        file!(), line!() as Index,
                    ))?;
                    let ok = self.set_numeric_value(tag, v, allow_clobber, false)?;
                    if !ok {
                        throw!(
                            ExceptionKind::OPTION_INVALID,
                            "Error setting numeric value read from file.".to_string()
                        );
                    }
                }
                OptionType::OT_Integer => {
                    let v: Index = value.parse().map_err(|_| SolverException::new(
                        ExceptionKind::OPTION_INVALID,
                        format!("Option \"{tag}\": Integer value expected, but non-integer option value \"{value}\" found.\n"),
                        file!(), line!() as Index,
                    ))?;
                    let ok = self.set_integer_value(tag, v, allow_clobber, false)?;
                    if !ok {
                        throw!(
                            ExceptionKind::OPTION_INVALID,
                            "Error setting integer value read from option file.".to_string()
                        );
                    }
                }
                OptionType::OT_Unknown => {
                    throw!(
                        ExceptionKind::OPTION_INVALID,
                        format!("Option \"{tag}\" has unknown type.")
                    );
                }
            }
        } else {
            self.set_string_value(tag, value, allow_clobber, false)?;
        }
        Ok(())
    }

    /// Mirrors `PrintList`. One option per line: `name value # used N times`.
    pub fn print_list(&self) -> String {
        let mut out = String::new();
        out.push_str("                                    Name   Value           # times used\n");
        for (k, v) in &self.options {
            out.push_str(&format!(
                "{:>40} = {:<30} # {}\n",
                k,
                v.value,
                v.counter.get()
            ));
        }
        out
    }

    /// Mirrors `PrintUserOptions`.
    pub fn print_user_options(&self) -> String {
        let mut out = String::new();
        for (k, v) in &self.options {
            if v.dont_print {
                continue;
            }
            let used = if v.counter.get() > 0 {
                "used"
            } else {
                "notused"
            };
            out.push_str(&format!("{} {} ({})\n", k, v.value, used));
        }
        out
    }
}

/// Parse a number, allowing Fortran-style `d`/`D` exponents (matching
/// `IpOptionsList::ReadFromStream`).
fn parse_ipopt_number(s: &str) -> Option<Number> {
    let mut buf = String::with_capacity(s.len());
    for c in s.chars() {
        if c == 'd' || c == 'D' {
            buf.push('e');
        } else {
            buf.push(c);
        }
    }
    buf.trim().parse().ok()
}

/// Tokeniser matching `OptionsList::readnexttoken` semantics:
/// whitespace splits tokens; `#` introduces a line comment; double
/// quotes group whitespace into a single token.
struct Tokenizer<'a> {
    chars: std::str::Chars<'a>,
    peeked: Option<char>,
}

impl<'a> Tokenizer<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            chars: s.chars(),
            peeked: None,
        }
    }

    fn next_char(&mut self) -> Option<char> {
        self.peeked.take().or_else(|| self.chars.next())
    }

    fn next_token(&mut self) -> Result<Option<String>, SolverException> {
        let mut c = match self.next_char() {
            Some(c) => c,
            None => return Ok(None),
        };
        loop {
            if c.is_whitespace() { /* skip */
            } else if c == '#' {
                // skip until newline
                loop {
                    match self.next_char() {
                        Some('\n') | None => break,
                        _ => {}
                    }
                }
            } else {
                break;
            }
            c = match self.next_char() {
                Some(c) => c,
                None => return Ok(None),
            };
        }
        let inside_quotes = c == '"';
        let mut tok = String::new();
        if inside_quotes {
            c = match self.next_char() {
                Some(c) => c,
                None => throw!(
                    ExceptionKind::OPTION_INVALID,
                    "Unterminated quoted string in option file.".to_string()
                ),
            };
        }
        loop {
            if !inside_quotes && c.is_whitespace() {
                return Ok(Some(tok));
            }
            if inside_quotes && c == '"' {
                return Ok(Some(tok));
            }
            tok.push(c);
            c = match self.next_char() {
                Some(c) => c,
                None => {
                    if inside_quotes {
                        throw!(
                            ExceptionKind::OPTION_INVALID,
                            "Unterminated quoted string in option file.".to_string()
                        );
                    }
                    return Ok(Some(tok));
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_basic() -> Rc<RegisteredOptions> {
        let r = RegisteredOptions::new();
        r.set_registering_category("Test");
        r.add_lower_bounded_number_option("tol", "Convergence tolerance", 0.0, true, 1e-8, "")
            .unwrap();
        r.add_string_option(
            "linear_solver",
            "Linear solver",
            "mumps",
            &[("mumps", ""), ("feral", "")],
            "",
        )
        .unwrap();
        r.add_lower_bounded_integer_option("max_iter", "Maximum iterations", 0, 3000, "")
            .unwrap();
        r.add_bool_option("print_user_options", "", false, "")
            .unwrap();
        r
    }

    #[test]
    fn prefix_lookup_overrides() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        o.set_numeric_value("tol", 1e-6, true, false).unwrap();
        o.set_numeric_value("resto.tol", 1e-3, true, false).unwrap();
        let (v_main, _) = o.get_numeric_value("tol", "").unwrap();
        let (v_resto, _) = o.get_numeric_value("tol", "resto.").unwrap();
        let (v_other, _) = o.get_numeric_value("tol", "noprefix.").unwrap();
        assert!((v_main - 1e-6).abs() < 1e-20);
        assert!((v_resto - 1e-3).abs() < 1e-20);
        assert!((v_other - 1e-6).abs() < 1e-20);
    }

    #[test]
    fn defaults_returned_when_unset() {
        let reg = registry_with_basic();
        let o = OptionsList::with_registered(reg);
        let (v, found) = o.get_numeric_value("tol", "").unwrap();
        assert!((v - 1e-8).abs() < 1e-20);
        assert!(!found);
    }

    #[test]
    fn read_options_file_text() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        let opt_file = "
# A comment line
tol  1.0e-7
max_iter 500
linear_solver mumps
print_user_options yes
";
        o.read_from_str(opt_file, false).unwrap();
        assert_eq!(o.get_numeric_value("tol", "").unwrap().0, 1e-7);
        assert_eq!(o.get_integer_value("max_iter", "").unwrap().0, 500);
        assert_eq!(o.get_string_value("linear_solver", "").unwrap().0, "mumps");
        assert!(o.get_bool_value("print_user_options", "").unwrap().0);
    }

    #[test]
    fn fortran_d_exponent_accepted() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        o.read_from_str("tol 1.0d-9\n", false).unwrap();
        assert!((o.get_numeric_value("tol", "").unwrap().0 - 1e-9).abs() < 1e-30);
    }

    #[test]
    fn unknown_option_in_file_is_error() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        let err = o.read_from_str("nonsense_option 1.0\n", false).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OPTION_INVALID);
    }

    #[test]
    fn invalid_string_value_rejected() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        let err = o
            .set_string_value("linear_solver", "ma27", true, false)
            .unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OPTION_INVALID);
    }

    #[test]
    fn out_of_range_number_rejected() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        let err = o.set_numeric_value("tol", 0.0, true, false).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OPTION_INVALID);
    }

    #[test]
    fn enum_value_index() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        o.set_string_value("linear_solver", "feral", true, false)
            .unwrap();
        assert_eq!(o.get_enum_value("linear_solver", "").unwrap().0, 1);
    }

    #[test]
    fn get_value_increments_use_counter() {
        let reg = registry_with_basic();
        let mut o = OptionsList::with_registered(reg);
        o.set_numeric_value("tol", 1e-6, true, false).unwrap();
        let _ = o.get_numeric_value("tol", "").unwrap();
        let _ = o.get_numeric_value("tol", "").unwrap();
        let listing = o.print_list();
        assert!(listing.contains("# 2"));
    }
}
