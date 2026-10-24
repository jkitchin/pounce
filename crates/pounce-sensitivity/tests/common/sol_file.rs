//! Reading what an AMPL solver writes: the `.sol` file, and the barrier
//! parameter Ipopt prints at `print_level 8`.

use pounce_common::types::Number;

use super::outside_engine::Reported;

/// The contents of a `.sol` file this tests needs: one dual per row, one
/// primal value per variable, and the real-valued variable suffixes.
#[derive(Debug, Clone, Default)]
pub struct SolFile {
    pub duals: Vec<Number>,
    pub primals: Vec<Number>,
    /// `(name, entries)` per variable suffix, each entry `(index, value)`.
    pub var_suffixes: Vec<(String, Vec<(usize, Number)>)>,
}

impl SolFile {
    /// Parses the text of a `.sol` file.
    ///
    /// The layout is the solver's message, an `Options` line, the option
    /// count and that many option values, then four counts: the rows, the
    /// duals written, the variables and the primals written. The duals and
    /// the primals follow one per line, then `objno`, then any suffix
    /// blocks. A suffix block is `suffix kind count name_length table_length`,
    /// its name, then `count` lines of `index value`. Kind bit 0 set means a
    /// row suffix, clear a variable suffix.
    pub fn parse(text: &str) -> Result<Self, String> {
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        let mut i = lines
            .iter()
            .position(|l| *l == "Options")
            .ok_or("no Options line")?;
        let int = |s: &str| -> Result<usize, String> {
            s.parse::<usize>()
                .map_err(|e| format!("bad count {s:?}: {e}"))
        };
        let num = |s: &str| -> Result<Number, String> {
            s.parse::<Number>()
                .map_err(|e| format!("bad value {s:?}: {e}"))
        };
        i += 1;
        let n_opts = int(lines.get(i).ok_or("truncated after Options")?)?;
        i += 1 + n_opts;
        let mut counts = [0usize; 4];
        for c in &mut counts {
            *c = int(lines.get(i).ok_or("truncated in the counts")?)?;
            i += 1;
        }
        let [_m, n_duals, _n, n_primals] = counts;
        let mut sol = SolFile::default();
        for _ in 0..n_duals {
            sol.duals
                .push(num(lines.get(i).ok_or("truncated in the duals")?)?);
            i += 1;
        }
        for _ in 0..n_primals {
            sol.primals
                .push(num(lines.get(i).ok_or("truncated in the primals")?)?);
            i += 1;
        }
        while i < lines.len() {
            let fields: Vec<&str> = lines[i].split_whitespace().collect();
            i += 1;
            if fields.first() != Some(&"suffix") {
                continue;
            }
            if fields.len() < 4 {
                return Err(format!("short suffix header {fields:?}"));
            }
            let kind = int(fields[1])?;
            let count = int(fields[2])?;
            let name = lines
                .get(i)
                .ok_or("truncated at a suffix name")?
                .to_string();
            i += 1;
            let mut entries = Vec::with_capacity(count);
            for _ in 0..count {
                let f: Vec<&str> = lines
                    .get(i)
                    .ok_or("truncated in a suffix")?
                    .split_whitespace()
                    .collect();
                i += 1;
                if f.len() != 2 {
                    return Err(format!("bad suffix entry {f:?}"));
                }
                entries.push((int(f[0])?, num(f[1])?));
            }
            if kind & 3 == 0 {
                sol.var_suffixes.push((name, entries));
            }
        }
        Ok(sol)
    }

    fn var_suffix(&self, name: &str, n: usize) -> Vec<Number> {
        let mut out = vec![0.0; n];
        if let Some((_, entries)) = self.var_suffixes.iter().find(|(s, _)| s == name) {
            for &(k, v) in entries {
                out[k] = v;
            }
        }
        out
    }

    /// The iterate in the sign convention the KKT residual uses.
    ///
    /// AMPL reports a row's dual with the opposite sign to Ipopt's `λ` in
    /// `∇f + Jᵀλ`, and Ipopt writes `ipopt_zU_out` negated, so both are
    /// flipped here. `ipopt_zL_out` is already the multiplier.
    pub fn ipopt_iterate(&self, mu: Number) -> Reported {
        let n = self.primals.len();
        Reported {
            x: self.primals.clone(),
            z_l: self.var_suffix("ipopt_zL_out", n),
            z_u: self
                .var_suffix("ipopt_zU_out", n)
                .iter()
                .map(|v| -v)
                .collect(),
            lambda: self.duals.iter().map(|v| -v).collect(),
            mu,
        }
    }
}

/// The barrier parameter of Ipopt's last iteration, from its output at
/// `print_level 8`: the last `Current barrier parameter mu = ...` line.
pub fn ipopt_final_mu(output: &str) -> Option<Number> {
    output
        .lines()
        .rev()
        .find_map(|l| l.trim().strip_prefix("Current barrier parameter mu ="))
        .and_then(|v| v.trim().parse().ok())
}

/// The factor Ipopt multiplied the objective by, from its output at
/// `print_level 8`: the `Scaling parameter for objective function = ...`
/// line. Ipopt prints it with seven significant digits, which is the
/// accuracy the engine built from it reproduces Ipopt's frame to.
pub fn ipopt_objective_scaling(output: &str) -> Option<Number> {
    output
        .lines()
        .rev()
        .find_map(|l| {
            l.trim()
                .strip_prefix("Scaling parameter for objective function =")
        })
        .and_then(|v| v.trim().parse().ok())
}
