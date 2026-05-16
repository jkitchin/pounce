//! Registered options registry.
//!
//! Mirrors `Common/IpRegOptions.{hpp,cpp}`. Each option that the
//! solver consults must be registered first with type, default, valid
//! range, and description. The registry is shared (via `Rc`) by the
//! `OptionsList` and by code that prints help.
//!
//! Insertion order is preserved (each option carries a `counter`), so
//! generated help text can be byte-identical to upstream when the
//! same registration order is used.

use crate::exception::{ExceptionKind, SolverException};
use crate::throw;
use crate::types::{Index, Number};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// Mirrors `RegisteredOptionType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum OptionType {
    OT_Number,
    OT_Integer,
    OT_String,
    OT_Unknown,
}

/// Mirrors `RegisteredOption::string_entry`.
#[derive(Debug, Clone)]
pub struct StringEntry {
    pub value: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum DefaultValue {
    None,
    Number(Number),
    Integer(Index),
    String(String),
}

/// Mirrors `RegisteredOption`. Holds metadata for one option.
#[derive(Debug, Clone)]
pub struct RegisteredOption {
    pub name: String,
    pub short_description: String,
    pub long_description: String,
    pub category: String,
    pub counter: Index,
    pub advanced: bool,
    pub option_type: OptionType,
    pub default: DefaultValue,
    pub has_lower: bool,
    pub lower: Number,
    pub lower_strict: bool,
    pub has_upper: bool,
    pub upper: Number,
    pub upper_strict: bool,
    pub valid_strings: Vec<StringEntry>,
}

impl RegisteredOption {
    fn new(name: String, short: String, long: String, category: String, counter: Index, advanced: bool) -> Self {
        Self {
            name,
            short_description: short,
            long_description: long,
            category,
            counter,
            advanced,
            option_type: OptionType::OT_Unknown,
            default: DefaultValue::None,
            has_lower: false,
            lower: 0.0,
            lower_strict: false,
            has_upper: false,
            upper: 0.0,
            upper_strict: false,
            valid_strings: Vec::new(),
        }
    }

    /// Equivalent to `IsValidNumberSetting` — checks bounds.
    pub fn is_valid_number(&self, v: Number) -> bool {
        if self.has_lower {
            let ok = if self.lower_strict { v > self.lower } else { v >= self.lower };
            if !ok { return false; }
        }
        if self.has_upper {
            let ok = if self.upper_strict { v < self.upper } else { v <= self.upper };
            if !ok { return false; }
        }
        true
    }

    pub fn is_valid_integer(&self, v: Index) -> bool {
        self.is_valid_number(v as Number)
    }

    /// Equivalent to `IsValidStringSetting`. A registered entry of
    /// `"*"` is treated as a wildcard: any string is accepted. This
    /// mirrors upstream Ipopt's behavior for free-form options like
    /// `output_file` and `linear_solver_options`.
    pub fn is_valid_string(&self, value: &str) -> bool {
        let v = value.to_ascii_lowercase();
        self.valid_strings
            .iter()
            .any(|e| e.value == "*" || e.value.eq_ignore_ascii_case(&v))
    }

    /// Returns the canonical (lowercase) form recorded at registration
    /// for the given enum value, or `None` if not allowed.
    pub fn canonical_string(&self, value: &str) -> Option<&str> {
        self.valid_strings
            .iter()
            .find(|e| e.value.eq_ignore_ascii_case(value))
            .map(|e| e.value.as_str())
    }

    /// Index of `value` in `valid_strings`, used for `GetEnumValue`.
    pub fn map_string_to_enum(&self, value: &str) -> Option<Index> {
        self.valid_strings
            .iter()
            .position(|e| e.value.eq_ignore_ascii_case(value))
            .map(|i| i as Index)
    }
}

/// Mirrors `RegisteredOptions`. Insertion-ordered registry of options.
#[derive(Debug, Default)]
pub struct RegisteredOptions {
    /// All registered options, keyed on lowercase name.
    options: RefCell<BTreeMap<String, Rc<RegisteredOption>>>,
    /// Insertion order used for printing.
    order: RefCell<Vec<String>>,
    /// Active category for `add_*` calls — set with `set_registering_category`.
    current_category: RefCell<String>,
    next_counter: RefCell<Index>,
}

impl RegisteredOptions {
    pub fn new() -> Rc<Self> { Rc::new(Self::default()) }

    pub fn set_registering_category(&self, category: impl Into<String>) {
        *self.current_category.borrow_mut() = category.into();
    }

    fn alloc_counter(&self) -> Index {
        let mut c = self.next_counter.borrow_mut();
        let v = *c;
        *c += 1;
        v
    }

    fn register(&self, opt: RegisteredOption) -> Result<Rc<RegisteredOption>, SolverException> {
        let key = opt.name.to_ascii_lowercase();
        let mut opts = self.options.borrow_mut();
        if opts.contains_key(&key) {
            throw!(
                ExceptionKind::OPTION_ALREADY_REGISTERED,
                format!("Option {} already registered.", opt.name)
            );
        }
        let rc = Rc::new(opt);
        opts.insert(key.clone(), rc.clone());
        self.order.borrow_mut().push(key);
        Ok(rc)
    }

    pub fn add_number_option(
        &self,
        name: &str,
        short_description: &str,
        default_value: Number,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(),
            short_description.to_string(),
            long_description.to_string(),
            self.current_category.borrow().clone(),
            self.alloc_counter(),
            false,
        );
        o.option_type = OptionType::OT_Number;
        o.default = DefaultValue::Number(default_value);
        self.register(o)
    }

    pub fn add_lower_bounded_number_option(
        &self,
        name: &str,
        short_description: &str,
        lower: Number,
        strict: bool,
        default_value: Number,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(), short_description.to_string(), long_description.to_string(),
            self.current_category.borrow().clone(), self.alloc_counter(), false,
        );
        o.option_type = OptionType::OT_Number;
        o.default = DefaultValue::Number(default_value);
        o.has_lower = true; o.lower = lower; o.lower_strict = strict;
        self.register(o)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_bounded_number_option(
        &self,
        name: &str,
        short_description: &str,
        lower: Number,
        lower_strict: bool,
        upper: Number,
        upper_strict: bool,
        default_value: Number,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(), short_description.to_string(), long_description.to_string(),
            self.current_category.borrow().clone(), self.alloc_counter(), false,
        );
        o.option_type = OptionType::OT_Number;
        o.default = DefaultValue::Number(default_value);
        o.has_lower = true; o.lower = lower; o.lower_strict = lower_strict;
        o.has_upper = true; o.upper = upper; o.upper_strict = upper_strict;
        self.register(o)
    }

    pub fn add_integer_option(
        &self,
        name: &str,
        short_description: &str,
        default_value: Index,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(), short_description.to_string(), long_description.to_string(),
            self.current_category.borrow().clone(), self.alloc_counter(), false,
        );
        o.option_type = OptionType::OT_Integer;
        o.default = DefaultValue::Integer(default_value);
        self.register(o)
    }

    pub fn add_lower_bounded_integer_option(
        &self,
        name: &str,
        short_description: &str,
        lower: Index,
        default_value: Index,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(), short_description.to_string(), long_description.to_string(),
            self.current_category.borrow().clone(), self.alloc_counter(), false,
        );
        o.option_type = OptionType::OT_Integer;
        o.default = DefaultValue::Integer(default_value);
        o.has_lower = true; o.lower = lower as Number;
        self.register(o)
    }

    pub fn add_bounded_integer_option(
        &self,
        name: &str,
        short_description: &str,
        lower: Index,
        upper: Index,
        default_value: Index,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(), short_description.to_string(), long_description.to_string(),
            self.current_category.borrow().clone(), self.alloc_counter(), false,
        );
        o.option_type = OptionType::OT_Integer;
        o.default = DefaultValue::Integer(default_value);
        o.has_lower = true; o.lower = lower as Number;
        o.has_upper = true; o.upper = upper as Number;
        self.register(o)
    }

    pub fn add_string_option(
        &self,
        name: &str,
        short_description: &str,
        default_value: &str,
        valid: &[(&str, &str)],
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        let mut o = RegisteredOption::new(
            name.to_string(), short_description.to_string(), long_description.to_string(),
            self.current_category.borrow().clone(), self.alloc_counter(), false,
        );
        o.option_type = OptionType::OT_String;
        o.default = DefaultValue::String(default_value.to_string());
        o.valid_strings = valid.iter()
            .map(|(v, d)| StringEntry { value: v.to_string(), description: d.to_string() })
            .collect();
        self.register(o)
    }

    /// Convenience: yes/no option, default `default_yes` ? "yes" : "no".
    pub fn add_bool_option(
        &self,
        name: &str,
        short_description: &str,
        default_yes: bool,
        long_description: &str,
    ) -> Result<Rc<RegisteredOption>, SolverException> {
        self.add_string_option(
            name,
            short_description,
            if default_yes { "yes" } else { "no" },
            &[("no", ""), ("yes", "")],
            long_description,
        )
    }

    /// Mirrors `GetOption(name)`. If `name` contains a `.`, only the
    /// suffix after the last `.` is looked up — this is how upstream
    /// validates prefixed option-file lines like `resto.tol`.
    pub fn get_option(&self, name: &str) -> Option<Rc<RegisteredOption>> {
        let tag_only = match name.rfind('.') {
            Some(pos) => &name[pos + 1..],
            None => name,
        };
        self.options.borrow().get(&tag_only.to_ascii_lowercase()).cloned()
    }

    /// Returns options in registration order.
    pub fn registered_options_in_order(&self) -> Vec<Rc<RegisteredOption>> {
        let opts = self.options.borrow();
        self.order
            .borrow()
            .iter()
            .filter_map(|k| opts.get(k).cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup_case_insensitive() {
        let r = RegisteredOptions::new();
        r.set_registering_category("Test");
        r.add_number_option("Tol", "tolerance", 1e-8, "").unwrap();
        assert!(r.get_option("tol").is_some());
        assert!(r.get_option("TOL").is_some());
    }

    #[test]
    fn duplicate_registration_is_error() {
        let r = RegisteredOptions::new();
        r.add_number_option("alpha", "", 1.0, "").unwrap();
        let err = r.add_number_option("ALPHA", "", 2.0, "").unwrap_err();
        assert_eq!(err.kind, ExceptionKind::OPTION_ALREADY_REGISTERED);
    }

    #[test]
    fn bounds_check_on_number() {
        let r = RegisteredOptions::new();
        r.add_lower_bounded_number_option("mu", "", 0.0, true, 0.1, "").unwrap();
        let opt = r.get_option("mu").unwrap();
        assert!(opt.is_valid_number(1e-12));
        assert!(!opt.is_valid_number(0.0));
        assert!(!opt.is_valid_number(-1.0));
    }

    #[test]
    fn string_enum_lookup() {
        let r = RegisteredOptions::new();
        r.add_string_option(
            "linear_solver", "",
            "mumps",
            &[("mumps", "MUMPS"), ("feral", "FERAL")],
            "",
        ).unwrap();
        let opt = r.get_option("linear_solver").unwrap();
        assert!(opt.is_valid_string("MuMpS"));
        assert!(!opt.is_valid_string("ma27"));
        assert_eq!(opt.map_string_to_enum("feral"), Some(1));
    }

    #[test]
    fn registration_order_preserved() {
        let r = RegisteredOptions::new();
        r.add_number_option("c", "", 0.0, "").unwrap();
        r.add_number_option("a", "", 0.0, "").unwrap();
        r.add_number_option("b", "", 0.0, "").unwrap();
        let order: Vec<_> = r.registered_options_in_order().iter().map(|o| o.name.clone()).collect();
        assert_eq!(order, vec!["c", "a", "b"]);
    }
}
