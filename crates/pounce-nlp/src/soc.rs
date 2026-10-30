//! Declared second-order-cone constraints over constraint rows (workstream C,
//! `dev-notes/research/nlp-soc-design.md`).
//!
//! A [`SocRows`] says `(g_{r0}(x), g_{r1}(x), …) ∈ K`, `K = { u : u₀ ≥
//! ‖u_{1..}‖ }`. The rows are declared **free** (`g_l = −∞`, `g_u = +∞`):
//! the cone is their only restriction. The IPM classifies a free row as an
//! inequality row `d(x) − s = 0` with no bounds on `s`, and carries the cone
//! on that block of slacks; [`resolve`] validates a declaration and maps it
//! into that `d` index space.
//!
//! A declaration is never dropped: anything that cannot honour it refuses
//! with the reason. A cone row read as a plain free row is an unconstrained
//! row — a wrong answer, not a degraded one.

use pounce_common::types::Index;

/// One second-order cone over constraint rows, in the model's own (0-based)
/// row indices; `rows[0]` is the cone's head `u₀`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocRows {
    pub rows: Vec<usize>,
}

/// Validate `cones` against the model's row bounds and map each cone's rows
/// to `d` indices (`full_to_d`, from the adapter's bound classification;
/// `-1` for a row that is not a `d` row). Refuses, with the reason: a cone
/// of fewer than two rows, a row index out of range, a row in two cones or
/// twice in one, and a row that is not free — a bound on a cone row is
/// either redundant with the cone or contradicts it, and which one depends
/// on where the row sits in the cone, so it is the model's to state as a
/// separate row.
pub fn resolve(
    cones: &[SocRows],
    g_l: &[f64],
    g_u: &[f64],
    lower_inf: f64,
    upper_inf: f64,
    full_to_d: &[Index],
) -> Result<Vec<Vec<usize>>, String> {
    let m = g_l.len();
    let mut owner: Vec<Option<usize>> = vec![None; m];
    let mut out = Vec::with_capacity(cones.len());
    for (k, cone) in cones.iter().enumerate() {
        if cone.rows.len() < 2 {
            return Err(format!(
                "second-order cone {k} has {} row(s); a cone needs a head and at least one \
                 more row (a one-row cone is a bound: write it as `g >= 0`)",
                cone.rows.len()
            ));
        }
        let mut d_rows = Vec::with_capacity(cone.rows.len());
        for &r in &cone.rows {
            if r >= m {
                return Err(format!(
                    "second-order cone {k} names row {r}, but the model has {m} rows"
                ));
            }
            if let Some(j) = owner[r] {
                return Err(if j == k {
                    format!("second-order cone {k} names row {r} twice")
                } else {
                    format!("row {r} is in second-order cones {j} and {k}; a row can be in one")
                });
            }
            owner[r] = Some(k);
            if g_l[r] > lower_inf || g_u[r] < upper_inf {
                return Err(format!(
                    "row {r} of second-order cone {k} has bounds [{}, {}]; cone rows must be \
                     free (-inf, +inf): the cone is their constraint",
                    g_l[r], g_u[r]
                ));
            }
            let d = full_to_d[r];
            if d < 0 {
                return Err(format!(
                    "row {r} of second-order cone {k} is not an inequality row of the \
                     classified problem"
                ));
            }
            d_rows.push(d as usize);
        }
        out.push(d_rows);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f64 = 1e19;

    fn cones(v: &[&[usize]]) -> Vec<SocRows> {
        v.iter().map(|r| SocRows { rows: r.to_vec() }).collect()
    }

    /// Rows 0 and 3 are equalities (c rows), 1, 2, 4 free (d rows 0, 1, 2).
    fn setup() -> (Vec<f64>, Vec<f64>, Vec<Index>) {
        (
            vec![1.0, -INF, -INF, 0.0, -INF],
            vec![1.0, INF, INF, 0.0, INF],
            vec![-1, 0, 1, -1, 2],
        )
    }

    #[test]
    fn free_rows_map_to_d_indices_head_first() {
        let (l, u, f) = setup();
        let got = resolve(&cones(&[&[4, 1, 2]]), &l, &u, -INF, INF, &f).unwrap();
        assert_eq!(got, vec![vec![2, 0, 1]]);
    }

    #[test]
    fn every_malformed_declaration_is_refused_with_its_reason() {
        let (l, u, f) = setup();
        let r = |c: &[&[usize]]| resolve(&cones(c), &l, &u, -INF, INF, &f).unwrap_err();
        assert!(r(&[&[1]]).contains("a cone needs a head"));
        assert!(r(&[&[1, 9]]).contains("model has 5 rows"));
        assert!(r(&[&[1, 1]]).contains("twice"));
        assert!(r(&[&[1, 2], &[4, 2]]).contains("cones 0 and 1"));
        assert!(r(&[&[0, 1]]).contains("must be free"));
    }

    #[test]
    fn a_one_sided_row_is_not_a_cone_row() {
        let (mut l, u, f) = setup();
        l[2] = 0.0;
        let e = resolve(&cones(&[&[1, 2]]), &l, &u, -INF, INF, &f).unwrap_err();
        assert!(e.contains("row 2") && e.contains("must be free"), "{e}");
    }
}
