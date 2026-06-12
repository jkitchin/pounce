//! Pre-flight problem inspection: builtin metadata, AMPL `.nl` header
//! parsing, GAMS `.gms` header / Solve-directive parsing, and `.lst`
//! SOLVE SUMMARY extraction.
//!
//! Ported from `studio/mcp/pounce_studio_mcp/server.py`. All entry
//! points are byte-slice / `&str` based so the module stays WASM-clean.

use serde::Serialize;

// ---- builtin problems ------------------------------------------------

/// Metadata for a CLI `--problem <name>` builtin.
#[derive(Debug, Clone, Serialize)]
pub struct BuiltinInfo {
    pub name: &'static str,
    pub n_variables: i32,
    pub n_constraints: i32,
    pub class: &'static str,
    pub notes: &'static str,
}

const BUILTINS: &[BuiltinInfo] = &[
    BuiltinInfo {
        name: "quadratic",
        n_variables: 2,
        n_constraints: 0,
        class: "unconstrained quadratic",
        notes: "Convex QP; trivial — single Newton step from any start.",
    },
    BuiltinInfo {
        name: "rosenbrock",
        n_variables: 2,
        n_constraints: 0,
        class: "unconstrained nonlinear",
        notes: "Classic non-convex banana valley; tests line search.",
    },
    BuiltinInfo {
        name: "bounded-quadratic",
        n_variables: 2,
        n_constraints: 0,
        class: "bound-constrained quadratic",
        notes: "Active-set quadratic; exercises bound multipliers.",
    },
    BuiltinInfo {
        name: "eq-quadratic",
        n_variables: 3,
        n_constraints: 1,
        class: "equality-constrained quadratic",
        notes: "QP with one linear equality; tests KKT factorisation.",
    },
    BuiltinInfo {
        name: "circle",
        n_variables: 2,
        n_constraints: 1,
        class: "equality-constrained nonlinear",
        notes: "Nonlinear equality; tests restoration entry.",
    },
];

/// Look up a builtin by name.
pub fn builtin(name: &str) -> Option<&'static BuiltinInfo> {
    BUILTINS.iter().find(|b| b.name == name)
}

/// All builtin names.
pub fn all_builtins() -> Vec<&'static BuiltinInfo> {
    BUILTINS.iter().collect()
}

// ---- option suggestions ---------------------------------------------

/// One advisory option suggestion. Never auto-applied.
#[derive(Debug, Clone, Serialize)]
pub struct Suggestion {
    pub option: String,
    pub value: String,
    pub why: String,
}

// ---- NL header parsing -----------------------------------------------

/// Dimensions / format detected from an AMPL `.nl` file header.
#[derive(Debug, Clone, Default, Serialize)]
pub struct NlHeader {
    pub format: String,
    pub n_variables: Option<i32>,
    pub n_constraints: Option<i32>,
    pub n_objectives: Option<i32>,
    pub n_ranges: Option<i32>,
    pub n_equalities: Option<i32>,
    pub n_nonlinear_constraints: Option<i32>,
    pub n_nonlinear_objectives: Option<i32>,
    pub n_nonlinear_vars_in_cons: Option<i32>,
    pub n_nonlinear_vars_in_obj: Option<i32>,
    pub n_nonlinear_vars_in_both: Option<i32>,
    pub nnz_jacobian: Option<i32>,
    pub nnz_objective_gradient: Option<i32>,
    pub warnings: Vec<String>,
}

/// Parse the first ~10 lines of an AMPL `.nl` file. Tolerant — partial
/// parses still return what we got.
pub fn parse_nl_header(bytes: &[u8]) -> NlHeader {
    let text = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = text.lines().take(10).collect();

    let mut out = NlHeader::default();
    if lines.is_empty() || lines[0].is_empty() {
        out.format = "unknown".into();
        out.warnings.push("empty .nl file".into());
        return out;
    }
    out.format = match lines[0].as_bytes().first() {
        Some(b'g') => "text".into(),
        Some(b'b') => "binary".into(),
        _ => "unknown".into(),
    };
    if out.format == "binary" {
        out.warnings.push("binary .nl: header parse skipped".into());
        return out;
    }

    let ints = |line: &str| -> Vec<i32> {
        line.split_whitespace()
            .filter_map(|t| t.parse::<i32>().ok())
            .collect()
    };

    if let Some(line) = lines.get(1) {
        let v = ints(line);
        if v.len() >= 5 {
            out.n_variables = Some(v[0]);
            out.n_constraints = Some(v[1]);
            out.n_objectives = Some(v[2]);
            out.n_ranges = Some(v[3]);
            out.n_equalities = Some(v[4]);
        } else {
            out.warnings.push("could not parse dimensions line".into());
        }
    }
    if let Some(line) = lines.get(2) {
        let v = ints(line);
        if v.len() >= 2 {
            out.n_nonlinear_constraints = Some(v[0]);
            out.n_nonlinear_objectives = Some(v[1]);
        }
    }
    if let Some(line) = lines.get(4) {
        let v = ints(line);
        if v.len() >= 3 {
            out.n_nonlinear_vars_in_cons = Some(v[0]);
            out.n_nonlinear_vars_in_obj = Some(v[1]);
            out.n_nonlinear_vars_in_both = Some(v[2]);
        }
    }
    for idx in [6_usize, 7] {
        if let Some(line) = lines.get(idx) {
            let v = ints(line);
            if v.len() == 2 && out.nnz_jacobian.is_none() {
                out.nnz_jacobian = Some(v[0]);
                out.nnz_objective_gradient = Some(v[1]);
                break;
            }
        }
    }
    out
}

/// Result of analyzing an NL file or builtin.
#[derive(Debug, Clone, Serialize)]
pub struct NlAnalysis {
    pub kind: String,
    pub name: Option<String>,
    pub path: Option<String>,
    pub dimensions: serde_json::Value,
    pub class: String,
    pub notes: Option<String>,
    pub warnings: Vec<String>,
    pub suggestions: Vec<Suggestion>,
}

/// Build an analysis from an NL header.
pub fn analyze_nl(path: &str, header: NlHeader) -> NlAnalysis {
    let class = classify_nl(&header);
    let warnings = nl_warnings(&header);
    let suggestions = nl_suggestions(&header);
    NlAnalysis {
        kind: "nl_file".into(),
        name: None,
        path: Some(path.into()),
        dimensions: serde_json::to_value(&header).unwrap_or(serde_json::Value::Null),
        class,
        notes: None,
        warnings,
        suggestions,
    }
}

/// Build an analysis from a builtin name.
pub fn analyze_builtin(name: &str) -> Result<NlAnalysis, String> {
    let b = builtin(name).ok_or_else(|| {
        let names: Vec<&str> = BUILTINS.iter().map(|b| b.name).collect();
        format!("unknown builtin {name:?}; valid: {names:?}")
    })?;
    let dims = serde_json::json!({
        "n_variables": b.n_variables,
        "n_constraints": b.n_constraints,
    });
    let header = NlHeader {
        n_variables: Some(b.n_variables),
        n_constraints: Some(b.n_constraints),
        ..Default::default()
    };
    Ok(NlAnalysis {
        kind: "builtin".into(),
        name: Some(b.name.into()),
        path: None,
        dimensions: dims,
        class: b.class.into(),
        notes: Some(b.notes.into()),
        warnings: nl_warnings(&header),
        suggestions: nl_suggestions(&header),
    })
}

fn classify_nl(h: &NlHeader) -> String {
    let n_con = h.n_constraints.unwrap_or(0);
    let nlc = h.n_nonlinear_constraints.unwrap_or(0);
    let nlo = h.n_nonlinear_objectives.unwrap_or(0);
    let n_eq = h.n_equalities.unwrap_or(0);
    let is_nl = nlc > 0 || nlo > 0;
    if n_con == 0 {
        return if is_nl {
            "unconstrained nonlinear".into()
        } else {
            "unconstrained linear/quadratic".into()
        };
    }
    let nl_or_lin = if is_nl {
        "nonlinear"
    } else {
        "linear/quadratic"
    };
    let eq_or_gen = if n_eq == n_con {
        "equality-constrained"
    } else {
        "general-constrained"
    };
    format!("{nl_or_lin} {eq_or_gen}")
}

fn nl_warnings(h: &NlHeader) -> Vec<String> {
    let mut out = h.warnings.clone();
    let n_var = h.n_variables.unwrap_or(0);
    let n_con = h.n_constraints.unwrap_or(0);
    if n_var == 0 {
        out.push("zero variables parsed — header read may have failed".into());
    }
    if (n_var + n_con) > 50_000 {
        out.push(format!(
            "very large problem ({n_var} vars, {n_con} cons); expect long solve times \
             and consider running with `--dump` for diagnostics.",
        ));
    }
    if h.n_objectives == Some(0) {
        out.push("no objective: this is a feasibility problem, not optimisation.".into());
    }
    out
}

fn nl_suggestions(h: &NlHeader) -> Vec<Suggestion> {
    let mut out = Vec::new();
    let n_var = h.n_variables.unwrap_or(0);
    let n_con = h.n_constraints.unwrap_or(0);
    let nlc = h.n_nonlinear_constraints.unwrap_or(0);
    let nlo = h.n_nonlinear_objectives.unwrap_or(0);
    let n_eq = h.n_equalities.unwrap_or(0);
    let size = n_var + n_con;

    if size > 1_000 && nlc == 0 && nlo == 0 {
        out.push(Suggestion {
            option: "mu_strategy".into(),
            value: "adaptive".into(),
            why: "purely linear/quadratic — adaptive mu usually converges in fewer iters.".into(),
        });
    }
    if size > 10_000 {
        out.push(Suggestion {
            option: "max_iter".into(),
            value: "1000".into(),
            why: "default 3000 is fine but raise tol expectations for large problems.".into(),
        });
    }
    if nlc > 0 && n_eq == n_con && n_con > 0 {
        out.push(Suggestion {
            option: "bound_relax_factor".into(),
            value: "0".into(),
            why: "all constraints equality + nonlinear: relaxing bounds can blur the feasible \
                  manifold; setting to 0 keeps it sharp."
                .into(),
        });
    }
    out
}

// ---- GAMS .gms header parsing ---------------------------------------

/// Dimensions parsed from a `gams convert`-emitted `.gms` header.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GmsHeader {
    pub n_equations_total: Option<i32>,
    pub n_equality_eqs: Option<i32>,
    pub n_ge_eqs: Option<i32>,
    pub n_le_eqs: Option<i32>,
    pub n_variables_total: Option<i32>,
    pub n_continuous_vars: Option<i32>,
    pub n_binary_vars: Option<i32>,
    pub n_integer_vars: Option<i32>,
    pub nnz_total: Option<i32>,
    pub nnz_constant: Option<i32>,
    pub nnz_nonlinear: Option<i32>,
}

/// Parse the comment block emitted by `gams convert`. Lines look like:
///
/// ```text
/// *  Equation counts
/// *     Total       E       G       L       N       X
/// *       109     108       0       1       0       0
/// ```
pub fn parse_gms_convert_header(text: &str) -> GmsHeader {
    let mut out = GmsHeader::default();
    let star_lines: Vec<&str> = text.lines().filter(|l| l.starts_with('*')).collect();

    fn next_int_line<'a>(lines: &'a [&'a str], start: usize) -> Option<Vec<i32>> {
        let end = (start + 5).min(lines.len());
        for line in lines.iter().take(end).skip(start + 1) {
            let nums: Vec<i32> = line
                .trim_start_matches('*')
                .split_whitespace()
                .filter_map(|t| t.parse::<i32>().ok())
                .collect();
            if !nums.is_empty() {
                return Some(nums);
            }
        }
        None
    }

    for (i, line) in star_lines.iter().enumerate() {
        if line.contains("Equation counts") {
            if let Some(nums) = next_int_line(&star_lines, i) {
                if !nums.is_empty() {
                    out.n_equations_total = Some(nums[0]);
                }
                if nums.len() >= 2 {
                    out.n_equality_eqs = Some(nums[1]);
                }
                if nums.len() >= 3 {
                    out.n_ge_eqs = Some(nums[2]);
                }
                if nums.len() >= 4 {
                    out.n_le_eqs = Some(nums[3]);
                }
            }
        } else if line.contains("Variable counts") {
            if let Some(nums) = next_int_line(&star_lines, i) {
                if !nums.is_empty() {
                    out.n_variables_total = Some(nums[0]);
                }
                if nums.len() >= 2 {
                    out.n_continuous_vars = Some(nums[1]);
                }
                if nums.len() >= 3 {
                    out.n_binary_vars = Some(nums[2]);
                }
                if nums.len() >= 4 {
                    out.n_integer_vars = Some(nums[3]);
                }
            }
        } else if line.contains("Nonzero counts") {
            if let Some(nums) = next_int_line(&star_lines, i) {
                if !nums.is_empty() {
                    out.nnz_total = Some(nums[0]);
                }
                if nums.len() >= 2 {
                    out.nnz_constant = Some(nums[1]);
                }
                if nums.len() >= 3 {
                    out.nnz_nonlinear = Some(nums[2]);
                }
            }
        }
    }
    out
}

/// Parsed `Solve <model> using <TYPE> [minimizing|maximizing] <objvar>;` line.
#[derive(Debug, Clone, Serialize)]
pub struct GmsSolveDirective {
    pub model_name: String,
    pub model_type: String,
    pub direction: Option<String>,
    pub objective_var: Option<String>,
}

/// Hand-rolled parser for the `Solve` directive. Case-insensitive,
/// scans all lines for the first match.
pub fn parse_gms_solve_directive(text: &str) -> Option<GmsSolveDirective> {
    for line in text.lines() {
        let lc = line.to_ascii_lowercase();
        let trimmed = lc.trim_start();
        if !trimmed.starts_with("solve") {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            continue;
        }
        // First token: "Solve" (matches case-insensitively).
        if !tokens[0].eq_ignore_ascii_case("solve") {
            continue;
        }
        let model_name = tokens[1]
            .trim_end_matches(',')
            .trim_end_matches(';')
            .to_string();
        // Expect "using" at tokens[2].
        if !tokens[2].eq_ignore_ascii_case("using") {
            continue;
        }
        let model_type = tokens[3]
            .trim_end_matches(',')
            .trim_end_matches(';')
            .to_ascii_uppercase();

        let mut direction: Option<String> = None;
        let mut objective_var: Option<String> = None;
        if tokens.len() >= 6 {
            let d = tokens[4].to_ascii_lowercase();
            if d == "minimizing" || d == "maximizing" {
                direction = Some(d);
                // Strip trailing `;` from objective var.
                let mut v = tokens[5].to_string();
                if let Some(s) = v.strip_suffix(';') {
                    v = s.to_string();
                }
                objective_var = Some(v);
            }
        }
        return Some(GmsSolveDirective {
            model_name,
            model_type,
            direction,
            objective_var,
        });
    }
    None
}

/// Result of analyzing a `.gms` file.
#[derive(Debug, Clone, Serialize)]
pub struct GmsAnalysis {
    pub path: String,
    pub dimensions: GmsHeader,
    pub solve_directive: Option<GmsSolveDirective>,
    pub class: String,
    pub supported_by_pounce: Option<bool>,
    pub suggestions: Vec<Suggestion>,
    pub warnings: Vec<String>,
}

pub fn analyze_gms(path: &str, text: &str) -> GmsAnalysis {
    let dims = parse_gms_convert_header(text);
    let solve = parse_gms_solve_directive(text);
    let model_type = solve.as_ref().map(|s| s.model_type.as_str());

    let mut warnings = Vec::new();
    if dims.n_variables_total.is_none() && dims.n_equations_total.is_none() {
        warnings.push(
            "no `gams convert` header found — dimensions could not be parsed. \
             POUNCE will still solve the model; the suggestion list is conservative."
                .into(),
        );
    }
    if solve.is_none() {
        warnings.push("no `Solve` directive found in file — is this a complete model?".into());
    }
    if let Some(mt @ ("MINLP" | "MIP")) = model_type {
        warnings.push(format!(
            "model type {mt} is not supported by POUNCE (integer variables present).",
        ));
    }
    if dims.n_binary_vars.unwrap_or(0) > 0 || dims.n_integer_vars.unwrap_or(0) > 0 {
        warnings.push(
            "discrete variables present; POUNCE solves the continuous relaxation only.".into(),
        );
    }

    let supported = model_type.map(|t| matches!(t, "NLP" | "DNLP" | "RMINLP"));
    GmsAnalysis {
        path: path.into(),
        class: classify_gms(model_type, &dims),
        suggestions: suggest_gms(&dims, model_type),
        solve_directive: solve,
        dimensions: dims,
        supported_by_pounce: supported,
        warnings,
    }
}

fn classify_gms(model_type: Option<&str>, dims: &GmsHeader) -> String {
    let Some(mt) = model_type else {
        return "unknown".into();
    };
    let base = match mt {
        "NLP" => "nonlinear program (continuous)",
        "DNLP" => "non-differentiable NLP",
        "RMINLP" => "relaxed mixed-integer NLP",
        "MINLP" => "mixed-integer NLP",
        "LP" => "linear program",
        "MIP" => "mixed-integer linear",
        "QCP" => "quadratically constrained program",
        "CNS" => "constrained nonlinear system",
        _ => return format!("{mt} model"),
    };
    if matches!(mt, "NLP" | "DNLP") && dims.nnz_nonlinear == Some(0) {
        format!("{base} (linear in nonzero pattern — should solve trivially)")
    } else {
        base.to_string()
    }
}

fn suggest_gms(dims: &GmsHeader, model_type: Option<&str>) -> Vec<Suggestion> {
    let mut out = Vec::new();
    let nnl = dims.nnz_nonlinear.unwrap_or(0);
    let nnz_total = dims.nnz_total.unwrap_or(1);

    if let Some(mt @ ("MINLP" | "MIP")) = model_type {
        out.push(Suggestion {
            option: "(none)".into(),
            value: "".into(),
            why: format!(
                "model type is {mt}; POUNCE handles only NLP/DNLP/RMINLP. Either relax \
                 the integrality (RMINLP) or pick a different solver.",
            ),
        });
        return out;
    }

    out.push(Suggestion {
        option: "mu_strategy".into(),
        value: "adaptive".into(),
        why: "matches GAMS-IPOPT's effective default (optipopt.def). pounce's compile-time \
              default is `monotone`, which stalls some hard NLPs."
            .into(),
    });
    if nnl > 0 && (nnl as f64) > 0.5 * (nnz_total as f64) {
        out.push(Suggestion {
            option: "tol".into(),
            value: "1e-6".into(),
            why: "heavily nonlinear pattern: tightening below 1e-6 often leads to dual \
                  stagnation on degenerate KKT systems."
                .into(),
        });
    }
    out
}

// ---- GAMS .lst SOLVE SUMMARY parsing --------------------------------

/// Parsed fields from a GAMS listing's `S O L V E   S U M M A R Y` block.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LstSummary {
    pub model: Option<String>,
    pub objective_var: Option<String>,
    pub solver: Option<String>,
    pub from_line: Option<i32>,
    pub solver_status_code: Option<i32>,
    pub solver_status: Option<String>,
    pub model_status_code: Option<i32>,
    pub model_status: Option<String>,
    pub objective_value: Option<serde_json::Value>,
    pub resource_used_secs: Option<serde_json::Value>,
    pub resource_limit_secs: Option<f64>,
    pub iteration_count: Option<serde_json::Value>,
    pub iteration_limit: Option<i32>,
    pub evaluation_errors: Option<serde_json::Value>,
    pub solver_status_file: Option<String>,
}

/// Parse a GAMS `.lst` listing. Tolerant: missing fields stay None.
pub fn parse_lst_summary(text: &str) -> LstSummary {
    let mut out = LstSummary::default();

    for line in text.lines() {
        let trimmed = line.trim_start();
        // MODEL <name>  ... OBJECTIVE <var>
        if let Some(rest) = trimmed.strip_prefix_ignore_ascii_case_re("MODEL") {
            // Look for "MODEL <name> ... OBJECTIVE <var>"
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if !toks.is_empty() {
                out.model.get_or_insert(toks[0].to_string());
            }
            if let Some(idx) = toks
                .iter()
                .position(|t| t.eq_ignore_ascii_case("OBJECTIVE"))
            {
                if let Some(v) = toks.get(idx + 1) {
                    out.objective_var.get_or_insert(v.to_string());
                }
            }
        }
        if let Some(rest) = trimmed.strip_prefix_ignore_ascii_case_re("SOLVER") {
            // "SOLVER <name> FROM LINE <n>"
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if !toks.is_empty() && out.solver.is_none() {
                out.solver = Some(toks[0].to_string());
            }
            if let Some(idx) = toks.iter().position(|t| t.eq_ignore_ascii_case("LINE")) {
                if let Some(v) = toks.get(idx + 1) {
                    if let Ok(n) = v.parse::<i32>() {
                        out.from_line.get_or_insert(n);
                    }
                }
            }
        }
        // **** SOLVER STATUS  N  <text>
        if let Some(rest) = line.strip_prefix("**** SOLVER STATUS") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if let Some(code) = toks.first().and_then(|s| s.parse::<i32>().ok()) {
                out.solver_status_code = Some(code);
                if toks.len() > 1 {
                    out.solver_status = Some(toks[1..].join(" "));
                }
            }
        }
        if let Some(rest) = line.strip_prefix("**** MODEL STATUS") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if let Some(code) = toks.first().and_then(|s| s.parse::<i32>().ok()) {
                out.model_status_code = Some(code);
                if toks.len() > 1 {
                    out.model_status = Some(toks[1..].join(" "));
                }
            }
        }
        if let Some(rest) = line.strip_prefix("**** OBJECTIVE VALUE") {
            if let Some(v) = rest.split_whitespace().next() {
                let val = v
                    .parse::<f64>()
                    .map(serde_json::Value::from)
                    .unwrap_or_else(|_| serde_json::Value::String(v.into()));
                out.objective_value = Some(val);
            }
        }
        if let Some(rest) = trimmed.strip_prefix_ignore_ascii_case_re("RESOURCE USAGE, LIMIT") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if let Some(v) = toks.first() {
                let val = v
                    .parse::<f64>()
                    .map(serde_json::Value::from)
                    .unwrap_or_else(|_| serde_json::Value::String((*v).to_string()));
                out.resource_used_secs = Some(val);
            }
            if let Some(v) = toks.get(1) {
                if let Ok(n) = v.parse::<f64>() {
                    out.resource_limit_secs.get_or_insert(n);
                }
            }
        }
        if let Some(rest) = trimmed.strip_prefix_ignore_ascii_case_re("ITERATION COUNT, LIMIT") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if let Some(v) = toks.first() {
                let val = v
                    .parse::<i64>()
                    .map(serde_json::Value::from)
                    .unwrap_or_else(|_| serde_json::Value::String((*v).to_string()));
                out.iteration_count = Some(val);
            }
            if let Some(v) = toks.get(1) {
                if let Ok(n) = v.parse::<i32>() {
                    out.iteration_limit.get_or_insert(n);
                }
            }
        }
        if let Some(rest) = trimmed.strip_prefix_ignore_ascii_case_re("EVALUATION ERRORS") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if let Some(v) = toks.first() {
                let val = v
                    .parse::<i64>()
                    .map(serde_json::Value::from)
                    .unwrap_or_else(|_| serde_json::Value::String((*v).to_string()));
                out.evaluation_errors = Some(val);
            }
        }
    }

    // Embedded solver-status block. Two formats:
    //   (a) `=C ...` between SOLVER STATUS FILE LISTED BELOW / ABOVE
    //   (b) lines after `--- POUNCE` and before `---- ` / `EXECUTION TIME`
    let mut block: Vec<String> = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        if line.contains("SOLVER STATUS FILE LISTED BELOW") {
            in_block = true;
            continue;
        }
        if line.contains("SOLVER STATUS FILE LISTED ABOVE") {
            in_block = false;
            continue;
        }
        if in_block && line.starts_with("=C") {
            block.push(line[2..].trim_end().to_string());
        }
    }
    if block.is_empty() {
        let mut capturing = false;
        for line in text.lines() {
            if !capturing && line.starts_with("--- POUNCE") {
                capturing = true;
            }
            if capturing {
                if line.starts_with("---- ") || line.starts_with("EXECUTION TIME") {
                    break;
                }
                block.push(line.trim_end().to_string());
            }
        }
    }
    if !block.is_empty() {
        out.solver_status_file = Some(block.join("\n").trim_end().to_string());
    }

    out
}

trait StripPrefixCi {
    fn strip_prefix_ignore_ascii_case_re<'a>(&'a self, prefix: &str) -> Option<&'a str>;
}

impl StripPrefixCi for str {
    fn strip_prefix_ignore_ascii_case_re<'a>(&'a self, prefix: &str) -> Option<&'a str> {
        if self.len() < prefix.len() {
            return None;
        }
        let (head, tail) = self.split_at(prefix.len());
        if head.eq_ignore_ascii_case(prefix) {
            Some(tail)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nl_header_basic() {
        let bytes = b"g3 1 1 0  # problem foo\n 5 3 1 0 2  # vars cons obj range eq\n 1 1  # nlc nlo\nplaceholder\n 2 1 0  # nlvc nlvo nlvb\n";
        let h = parse_nl_header(bytes);
        assert_eq!(h.format, "text");
        assert_eq!(h.n_variables, Some(5));
        assert_eq!(h.n_constraints, Some(3));
        assert_eq!(h.n_nonlinear_constraints, Some(1));
    }

    #[test]
    fn nl_header_empty_file() {
        let h = parse_nl_header(b"");
        assert_eq!(h.format, "unknown");
        assert!(h.warnings.iter().any(|w| w.contains("empty")));
    }

    #[test]
    fn nl_header_binary_short_circuits() {
        let h = parse_nl_header(b"b3 1 1 0\nignored\n");
        assert_eq!(h.format, "binary");
    }

    #[test]
    fn gms_solve_directive_simple() {
        let text = "Variables x, z;\nEquation foo;\n\nSolve mymodel using NLP minimizing z;\n";
        let d = parse_gms_solve_directive(text).expect("should parse");
        assert_eq!(d.model_name, "mymodel");
        assert_eq!(d.model_type, "NLP");
        assert_eq!(d.direction.as_deref(), Some("minimizing"));
        assert_eq!(d.objective_var.as_deref(), Some("z"));
    }

    #[test]
    fn gms_solve_directive_lower_case() {
        let text = "solve hs071 using nlp minimizing obj ;\n";
        let d = parse_gms_solve_directive(text).expect("should parse");
        assert_eq!(d.model_type, "NLP");
        assert_eq!(d.direction.as_deref(), Some("minimizing"));
    }

    #[test]
    fn gms_convert_header_counts() {
        let text = "* Equation counts\n*    Total       E       G       L       N       X\n*       10       8       1       1       0       0\n* Variable counts\n*    Total    cont  binary integer    sos1    sos2   scont    sint\n*       12      11       0       1       0       0       0       0\n* Nonzero counts\n*    Total   const      NL     DLL\n*       30      20      10       0\n";
        let h = parse_gms_convert_header(text);
        assert_eq!(h.n_equations_total, Some(10));
        assert_eq!(h.n_equality_eqs, Some(8));
        assert_eq!(h.n_variables_total, Some(12));
        assert_eq!(h.n_integer_vars, Some(1));
        assert_eq!(h.nnz_nonlinear, Some(10));
    }

    #[test]
    fn lst_summary_parses_status() {
        let text = "                MODEL   m       OBJECTIVE  z\n                SOLVER  POUNCE  FROM LINE  10\n**** SOLVER STATUS     1 Normal Completion\n**** MODEL STATUS      2 Locally Optimal\n**** OBJECTIVE VALUE  3.14159\n RESOURCE USAGE, LIMIT  0.123  1000.000\n ITERATION COUNT, LIMIT  42  5000\n EVALUATION ERRORS  0  0\n";
        let s = parse_lst_summary(text);
        assert_eq!(s.solver_status_code, Some(1));
        assert_eq!(s.model_status_code, Some(2));
        assert_eq!(s.iteration_limit, Some(5000));
    }
}
