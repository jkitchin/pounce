//! Degeneracy of the equality constraints: structural, then numerical.
//!
//! * `structural_rank` — a maximum matching between equality rows and the
//!   (non-fixed) variables they touch, and its Dulmage–Mendelsohn
//!   decomposition (`pounce-presolve`). Unmatched equality rows mean the
//!   equality Jacobian is rank deficient at **every** point: the sparsity
//!   pattern bounds the rank from above. The over-determined block names the
//!   rows and variables involved.
//!
//! * `numerical_rank` — a dense SVD of the equality Jacobian at the
//!   evaluation point, under one absolute singular-value threshold. When it
//!   is rank deficient, a deletion filter shrinks the rows taking part in the
//!   left null space to an **inclusion-minimal dependent set**: a set that is
//!   dependent, while removing any one of its rows leaves an independent set.
//!   That is the "irreducible degenerate set" of Dowling & Biegler's
//!   *Degeneracy Hunter*, here computed numerically at one point and one
//!   threshold. The left-null combination is reported in original row names.
//!   If the deficiency is `d`, up to `max_dependent_sets` such sets are found
//!   by removing one row of each set found and searching again.
//!
//!   A numerical rank below the structural rank means the dependence comes
//!   from the values, not the pattern — it can come and go between points.

use super::report::{Basis, Confidence, Finding, Report, Severity, num};
use super::{DiagnoseOptions, PointEval, cname, vname};
use crate::nl_reader::NlProblem;
use faer::Mat;
use pounce_common::types::Number;
use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
use pounce_presolve::matching::hopcroft_karp;
use pounce_presolve::{DMPart, DulmageMendelsohnPartition};
use serde_json::json;
use std::collections::BTreeMap;

/// Rows of the left null space with participation below this are ignored.
const NULL_WEIGHT_TOL: Number = 1e-3;

/// Names listed in a finding's text before "and N more".
const NAMES_SHOWN: usize = 12;

pub(crate) fn list(names: &[String]) -> String {
    let shown: Vec<&str> = names.iter().take(NAMES_SHOWN).map(String::as_str).collect();
    let more = names.len().saturating_sub(shown.len());
    if more > 0 {
        format!("{} and {more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

pub(crate) fn fixed_mask(prob: &NlProblem) -> Vec<bool> {
    (0..prob.n).map(|j| prob.x_l[j] == prob.x_u[j]).collect()
}

/// Equality rows, in row order.
pub(crate) fn equality_rows(prob: &NlProblem) -> Vec<usize> {
    (0..prob.m)
        .filter(|&i| prob.g_l[i] == prob.g_u[i] && prob.g_l[i].is_finite())
        .collect()
}

/// `structural_rank`.
/// Returns the structural rank, for [`check_numerical`] to compare against.
pub fn check_structural(prob: &NlProblem, pt: &PointEval, report: &mut Report) -> Option<usize> {
    if !pt.jac_structure_ok {
        report.unavailable(
            "structural_rank",
            "the Jacobian sparsity structure could not be obtained",
        );
        return None;
    }
    let irow: Vec<i32> = pt.jac_irow.iter().map(|&r| r as i32).collect();
    let jcol: Vec<i32> = pt.jac_jcol.iter().map(|&c| c as i32).collect();
    let fixed = fixed_mask(prob);
    let inc = EqualityIncidence::from_probe(&ProbeView {
        n_vars: prob.n,
        m_rows: prob.m,
        jac_irow: &irow,
        jac_jcol: &jcol,
        jac_values: None,
        g_l: &prob.g_l,
        g_u: &prob.g_u,
        linearity: None,
        one_based: false,
        eq_tol: 0.0,
        excluded_vars: Some(&fixed),
        excluded_rows: None,
    });
    let n_eq = inc.n_eq_rows();
    let matching = hopcroft_karp(&inc);
    let dm = DulmageMendelsohnPartition::from_matching(&inc, &matching);
    let n_free = fixed.iter().filter(|f| !**f).count();
    report.sections.insert(
        "structural_rank",
        json!({
            "equality_rows": n_eq,
            "free_variables": n_free,
            "structural_rank": matching.size,
            "overdetermined_rows": dm.over_rows.len(),
            "underdetermined_variables": dm.under_cols.iter().filter(|&&j| !fixed[j]).count(),
            "degrees_of_freedom": n_free as i64 - matching.size as i64,
        }),
    );
    let mut sink = report.ran("structural_rank");
    if matching.size < n_eq {
        let rows: Vec<String> = dm
            .over_rows
            .iter()
            .filter(|&&k| dm.row_part[k] == DMPart::Over)
            .map(|&k| cname(prob, inc.eq_row_inner_idx[k]))
            .collect();
        let cols: Vec<String> = dm.over_cols.iter().map(|&j| vname(prob, j)).collect();
        let deficiency = n_eq - matching.size;
        sink.push(
            Finding::new("structural_rank", "structurally_overdetermined_equalities", Severity::Warning,
                Basis::Structural, Confidence::Certain,
                format!("{deficiency} of the {n_eq} equality rows cannot be matched to a distinct free variable. The over-determined block has {} equation(s) in {} variable(s): equations {}; variables {}.",
                    rows.len(), cols.len(), list(&rows), if cols.is_empty() { "(none)".to_string() } else { list(&cols) }))
            .why("The equality Jacobian is rank deficient at every point (its sparsity pattern alone caps the rank), so LICQ fails everywhere: the multipliers are not unique and the solver must regularize. If the extra equations are not consistent, the problem is infeasible.")
            .next("Look for an equation that is implied by the others (a balance written once too often), or a variable that should be free but is fixed.")
            .cons(rows)
            .vars(cols)
            .evidence(json!({"equality_rows": n_eq, "structural_rank": matching.size,
                "structural_deficiency": deficiency, "fixed_variables_excluded": n_free < prob.n})),
        );
    }
    Some(matching.size)
}

/// A dense row-major matrix of Jacobian rows (or other gradients) over a
/// selection of columns, with the identity of each row and column kept.
pub(crate) struct Dense {
    /// Original row identity of each dense row (meaning is up to the
    /// caller: a constraint index, or an encoded active-set entry).
    pub rows: Vec<usize>,
    /// Original variable index of each dense column.
    pub cols: Vec<usize>,
    /// `rows.len() × max(cols.len(), 1)` entries, row-major.
    pub a: Vec<Number>,
}

impl Dense {
    pub fn ncols(&self) -> usize {
        self.cols.len().max(1)
    }

    /// The submatrix of rows `keep`, restricted to the columns those rows
    /// touch (the other columns cannot change the rank).
    pub fn sub(&self, keep: &[usize]) -> Mat<Number> {
        let n = self.ncols();
        let used: Vec<usize> = (0..self.cols.len())
            .filter(|&c| keep.iter().any(|&r| self.a[r * n + c] != 0.0))
            .collect();
        Mat::from_fn(keep.len(), used.len().max(1), |i, j| {
            used.get(j).map(|&c| self.a[keep[i] * n + c]).unwrap_or(0.0)
        })
    }

    /// The whole matrix.
    pub fn full(&self) -> Mat<Number> {
        let n = self.ncols();
        Mat::from_fn(self.rows.len(), n, |i, j| self.a[i * n + j])
    }

    /// Singular values, largest first.
    pub fn singular_values(&self) -> Option<Vec<Number>> {
        if self.rows.is_empty() {
            return Some(Vec::new());
        }
        let mut s = self.full().singular_values().ok()?;
        s.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        Some(s)
    }
}

/// Why a dense block could not be built.
#[derive(Debug, Clone)]
pub(crate) enum DenseError {
    /// Larger than the dense limits: the sparse inertia count still works.
    TooLarge(String),
    /// Non-finite entries: no rank method applies.
    NonFinite(String),
}

impl std::fmt::Display for DenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DenseError::TooLarge(s) | DenseError::NonFinite(s) => f.write_str(s),
        }
    }
}

/// The sparse block of Jacobian rows `rows` (each scaled by `signs`, or 1)
/// over the columns `cols`, as 0-based `(row, col, value)` triplets in
/// block coordinates. Entries with a non-finite value make it `None`.
pub(crate) fn sparse_block(
    pt: &PointEval,
    m_total: usize,
    rows: &[usize],
    signs: Option<&[Number]>,
    cols: &[usize],
    n_total: usize,
) -> Option<Vec<(usize, usize, Number)>> {
    let mut row_pos = vec![usize::MAX; m_total];
    for (p, &r) in rows.iter().enumerate() {
        row_pos[r] = p;
    }
    let mut col_pos = vec![usize::MAX; n_total];
    for (p, &c) in cols.iter().enumerate() {
        col_pos[c] = p;
    }
    let mut out = Vec::new();
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if r >= m_total || c >= n_total || row_pos[r] == usize::MAX || col_pos[c] == usize::MAX {
            continue;
        }
        let v = pt.jac_val[k] * signs.map(|s| s[row_pos[r]]).unwrap_or(1.0);
        if !v.is_finite() {
            return None;
        }
        if v != 0.0 {
            out.push((row_pos[r], col_pos[c], v));
        }
    }
    Some(out)
}

/// Pivots at or below this, on feral's scaled `[[I, Jᵀ], [J, 0]]`, count as
/// zero in [`feral_rank`].
pub(crate) const FERAL_ZERO_TOL: Number = 1e-10;

/// Rank of a sparse `m × n` matrix `J`, read from the inertia of
/// `K = [[I, Jᵀ], [J, 0]]` as factored by feral (pounce's own sparse
/// symmetric-indefinite solver, at its default settings). The inertia of
/// `K` is `(n, rank J, m − rank J)` exactly, so the negative count is the
/// rank and the zero count the deficiency — the same inertia the
/// interior-point method checks on its KKT matrix every iteration.
///
/// What counts as zero is feral's pivot decision on the scaled matrix, not
/// a singular-value threshold. It is therefore insensitive to row scaling
/// (a tiny but independent row is not dependent), and coarser than the SVD
/// (rows that differ in the ninth digit count as dependent): "dependent
/// enough to trouble the solver", which is why it is a second opinion and
/// not a replacement. It gives a count only, no localization. Measured on
/// the 134-model fixture corpus at each solve's solution, it agrees with the
/// SVD on 127; it can also miss a dependency the SVD finds (1 model), and
/// it declines (an `Err`) when it cannot resolve some pivots' signs (3).
pub(crate) fn feral_rank(
    m: usize,
    n: usize,
    entries: &[(usize, usize, Number)],
) -> Result<usize, String> {
    if m == 0 {
        return Ok(0);
    }
    if n == 0 {
        return Ok(0);
    }
    let cap = n + entries.len() + m;
    let (mut r, mut c, mut v) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    for k in 0..n {
        r.push(k);
        c.push(k);
        v.push(1.0);
    }
    for &(i, j, a) in entries {
        r.push(n + i);
        c.push(j);
        v.push(a);
    }
    // Keep the (2,2) diagonal in the pattern so every row is present.
    for i in 0..m {
        r.push(n + i);
        c.push(n + i);
        v.push(0.0);
    }
    let k =
        feral::CscMatrix::from_triplets(n + m, &r, &c, &v).map_err(|e| format!("feral: {e}"))?;
    // Count pivots at or below `FERAL_ZERO_TOL` (on feral's scaled matrix)
    // as zero instead of by their round-off sign: with the default
    // `zero_tol = ε` a dependency's pivot of ~1e-14 is counted positive or
    // negative at random and the inertia identity breaks.
    let zero_tol = FERAL_ZERO_TOL;
    let mut np = feral::NumericParams::default();
    np.bk.zero_tol = zero_tol;
    np.bk.zero_tol_2x2 = zero_tol * zero_tol;
    np.bk.null_pivot_tol = zero_tol;
    np.bk.null_pivot_tol_2x2 = zero_tol * zero_tol;
    np.bk.on_zero_pivot = feral::ZeroPivotAction::ForceAccept;
    // Cascade-break floors tiny pivots to ±floor, which leaves their sign
    // (and so the inertia) undetermined; a rank count needs them counted.
    np.cascade_break_ratio = None;
    np.cascade_break_eps = None;
    // Static pivoting floors them too.
    np.static_pivot_threshold = None;
    np.bk.static_pivot_floor = 0.0;
    let mut solver = feral::Solver::with_params(np, Default::default());
    match solver.factor(&k, None) {
        feral::FactorStatus::Success | feral::FactorStatus::WrongInertia { .. } => {}
        other => {
            return Err(format!(
                "feral could not factor [[I, Jᵀ], [J, 0]]: {other:?}"
            ));
        }
    }
    let inertia = solver.inertia().ok_or("feral reported no inertia")?;
    // `n_tiny` also counts the pivots accepted as zero, which are exactly
    // the ones we want; only floored pivots beyond those were counted by a
    // round-off sign.
    if let Some(u) = solver
        .last_factor_stats()
        .map(|st| st.inertia_uncertain().saturating_sub(inertia.zero))
        && u > 0
    {
        return Err(format!(
            "feral could not resolve the sign of {u} pivot(s), so its inertia is not a rank"
        ));
    }
    if inertia.positive != n || inertia.negative + inertia.zero != m {
        return Err(format!(
            "feral's inertia ({}, {}, {}) is inconsistent with a {m} × {n} block",
            inertia.positive, inertia.negative, inertia.zero
        ));
    }
    Ok(inertia.negative)
}

/// The dense block of Jacobian rows `rows` at `pt`, over the columns of
/// non-fixed variables those rows touch. `Err` carries why it could not be
/// built (non-finite entries, or larger than the dense limits).
pub(crate) fn dense_jacobian_rows(
    prob: &NlProblem,
    pt: &PointEval,
    rows: &[usize],
    opts: &DiagnoseOptions,
) -> Result<Dense, DenseError> {
    let fixed = fixed_mask(prob);
    let mut row_pos = vec![usize::MAX; prob.m];
    for (p, &r) in rows.iter().enumerate() {
        row_pos[r] = p;
    }
    let mut col_pos: BTreeMap<usize, usize> = BTreeMap::new();
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if r < prob.m && row_pos[r] != usize::MAX && !fixed[c] {
            let next = col_pos.len();
            col_pos.entry(c).or_insert(next);
        }
    }
    let (m, n) = (rows.len(), col_pos.len());
    // Non-finite entries rule out every rank method; check them first.
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if r < prob.m && row_pos[r] != usize::MAX && !fixed[c] && !pt.jac_val[k].is_finite() {
            return Err(DenseError::NonFinite(format!(
                "the Jacobian has non-finite entries at {} (see nonfinite_localization)",
                pt.label
            )));
        }
    }
    if m.saturating_mul(n.max(1)) > opts.max_dense_entries || m.min(n) > opts.max_dense_dim {
        return Err(DenseError::TooLarge(format!(
            "the block is {m} × {n}, larger than the dense-SVD limit (--max-dense-entries {}, --max-dense-dim {})",
            opts.max_dense_entries, opts.max_dense_dim
        )));
    }
    let mut cols: Vec<usize> = vec![0; n];
    for (&c, &p) in &col_pos {
        cols[p] = c;
    }
    let ncols = n.max(1);
    let mut a = vec![0.0; m * ncols];
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if r >= prob.m || row_pos[r] == usize::MAX || fixed[c] {
            continue;
        }
        a[row_pos[r] * ncols + col_pos[&c]] += pt.jac_val[k];
    }
    Ok(Dense {
        rows: rows.to_vec(),
        cols,
        a,
    })
}

pub(crate) fn rank_of(m: &Mat<Number>, tau: Number) -> Option<usize> {
    let s = m.singular_values().ok()?;
    Some(s.iter().filter(|&&v| v > tau).count())
}

/// Left-null weights of `keep` rows: Σ over null columns of U of u².
fn null_weights(d: &Dense, keep: &[usize], tau: Number) -> Option<Vec<Number>> {
    let m = d.sub(keep);
    let svd = m.svd().ok()?;
    let s: Vec<Number> = svd.S().column_vector().iter().copied().collect();
    let rank = s.iter().filter(|&&v| v > tau).count();
    let u = svd.U();
    let mut w = vec![0.0; keep.len()];
    for k in rank..keep.len() {
        for (i, val) in u.col(k).iter().enumerate() {
            w[i] += val * val;
        }
    }
    Some(w)
}

/// The left-null combination of a minimal dependent set (one dependency),
/// scaled so its largest coefficient is ±1, and the relative residual
/// `‖cᵀ A_S‖∞ / max|A_S|`.
fn null_combination(d: &Dense, set: &[usize]) -> Option<(Vec<Number>, Number)> {
    let m = d.sub(set);
    let svd = m.svd().ok()?;
    let u = svd.U();
    let col = u.col(set.len() - 1);
    let mut c: Vec<Number> = col.iter().copied().collect();
    let scale = c.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if scale > 0.0 {
        for v in &mut c {
            *v /= scale;
        }
    }
    let ncols = m.ncols();
    let mut resid = 0.0_f64;
    let mut amax = 0.0_f64;
    for j in 0..ncols {
        let mut acc = 0.0;
        for i in 0..set.len() {
            acc += c[i] * m[(i, j)];
            amax = amax.max(m[(i, j)].abs());
        }
        resid = resid.max(acc.abs());
    }
    Some((c, if amax > 0.0 { resid / amax } else { 0.0 }))
}

/// One inclusion-minimal dependent set: dense row positions, the left-null
/// combination over them, and its relative residual.
pub(crate) struct DependentSet {
    pub rows: Vec<usize>,
    pub combination: Vec<Number>,
    pub residual: Number,
}

/// What [`minimal_dependent_sets`] found.
pub(crate) struct SetSearch {
    pub sets: Vec<DependentSet>,
    /// The candidate set that exceeded `max_search`, when the search stopped
    /// on it (dense row positions).
    pub too_large: Option<Vec<usize>>,
}

/// Localize up to `max_sets` inclusion-minimal dependent row sets of `d`
/// under the absolute singular-value threshold `tau`. Each set is shrunk
/// from the rows taking part in the left null space by a deletion filter
/// (least-involved rows first); after each set, the row carrying its
/// largest coefficient is removed and the search repeats.
pub(crate) fn minimal_dependent_sets(
    d: &Dense,
    tau: Number,
    deficiency: usize,
    max_sets: usize,
    max_search: usize,
) -> SetSearch {
    let mut out = SetSearch {
        sets: Vec::new(),
        too_large: None,
    };
    let mut working: Vec<usize> = (0..d.rows.len()).collect();
    while out.sets.len() < deficiency.min(max_sets) {
        let Some(w) = null_weights(d, &working, tau) else {
            break;
        };
        let mut cand: Vec<(usize, Number)> = working
            .iter()
            .zip(&w)
            .filter(|(_, wt)| **wt > NULL_WEIGHT_TOL)
            .map(|(&r, &wt)| (r, wt))
            .collect();
        let dependent = |set: &[usize]| rank_of(&d.sub(set), tau).map(|r| r < set.len());
        let mut set: Vec<usize> = cand.iter().map(|c| c.0).collect();
        if set.is_empty() || dependent(&set) != Some(true) {
            set = working.clone();
            cand = working.iter().map(|&r| (r, 0.0)).collect();
        }
        if set.len() > max_search {
            out.too_large = Some(set);
            break;
        }
        cand.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (r, _) in &cand {
            if set.len() <= 1 {
                break;
            }
            let trial: Vec<usize> = set.iter().copied().filter(|x| x != r).collect();
            if dependent(&trial) == Some(true) {
                set = trial;
            }
        }
        set.sort_unstable();
        let Some((coef, resid)) = null_combination(d, &set) else {
            break;
        };
        let drop = set
            .iter()
            .zip(&coef)
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&p, _)| p);
        out.sets.push(DependentSet {
            rows: set,
            combination: coef,
            residual: resid,
        });
        match drop {
            Some(p) => working.retain(|&x| x != p),
            None => break,
        }
    }
    out
}

/// The default relative singular-value threshold, `max(m, n)·ε`, or the
/// user's `--rank-tol`.
pub(crate) fn default_rel_tol(d: &Dense, opts: &DiagnoseOptions) -> Number {
    opts.rank_rel_tol
        .unwrap_or((d.rows.len().max(d.cols.len()) as Number) * Number::EPSILON)
}

/// `numerical_rank`.
pub fn check_numerical(
    prob: &NlProblem,
    pt: &PointEval,
    structural: Option<usize>,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    if !pt.jac_ok {
        report.unavailable(
            "numerical_rank",
            format!("the Jacobian could not be evaluated at {}", pt.label),
        );
        return;
    }
    let eq = equality_rows(prob);
    if eq.is_empty() {
        report.ran("numerical_rank");
        return;
    }
    let d = match dense_jacobian_rows(prob, pt, &eq, opts) {
        Ok(d) => d,
        Err(DenseError::TooLarge(why)) => {
            sparse_numerical_rank(prob, pt, &eq, structural, &why, report);
            return;
        }
        Err(e) => {
            report.unavailable("numerical_rank", format!("equality Jacobian: {e}"));
            return;
        }
    };
    let (m, n) = (d.rows.len(), d.cols.len());
    let Some(s) = d.singular_values() else {
        report.unavailable("numerical_rank", "the SVD did not converge");
        return;
    };
    let smax = s.first().copied().unwrap_or(0.0);
    let rel = default_rel_tol(&d, opts);
    let tau = smax * rel;
    report
        .tolerances
        .insert("rank_singular_value_threshold", tau);
    let rank = s.iter().filter(|&&v| v > tau).count();
    let smin_pos = s
        .iter()
        .copied()
        .filter(|&v| v > tau)
        .fold(Number::INFINITY, Number::min);
    report.sections.insert(
        "numerical_rank",
        json!({
            "point": pt.label,
            "equality_rows": m,
            "columns": n,
            "rank": rank,
            "deficiency": m.saturating_sub(rank),
            "sigma_max": num(smax),
            "smallest_retained_sigma": num(smin_pos),
            "threshold": num(tau),
            "relative_threshold": rel,
            "scaling": "none (unscaled Jacobian)",
            "structural_rank": structural,
        }),
    );

    let mut findings: Vec<Finding> = Vec::new();
    if let Some(sr) = structural
        && rank < sr
    {
        findings.push(
                Finding::new("numerical_rank", "numerical_rank_below_structural_rank", Severity::Warning,
                    Basis::Numerical, Confidence::High,
                    format!("At {}, the equality Jacobian has numerical rank {rank}, below its structural rank {sr}, so a deficiency of {} comes from the values, not the sparsity pattern.",
                        pt.label, sr - rank))
                .why("Value-dependent rank loss can appear at some points and not others; where it appears the solver needs dual regularization and the multipliers are not unique.")
                .next("Inspect the dependent sets below, and see rank_robustness for whether the loss persists at other points, tolerances and scalings.")
                .evidence(json!({"point": pt.label, "rank": rank, "structural_rank": sr, "threshold": num(tau)})),
            );
    }

    let deficiency = m.saturating_sub(rank);
    let search = minimal_dependent_sets(
        &d,
        tau,
        deficiency,
        opts.max_dependent_sets,
        opts.max_dependent_search,
    );
    for set in &search.sets {
        let names: Vec<String> = set.rows.iter().map(|&p| cname(prob, d.rows[p])).collect();
        let combo: Vec<serde_json::Value> = set
            .rows
            .iter()
            .zip(&set.combination)
            .map(|(&p, &c)| json!({"row": cname(prob, d.rows[p]), "coefficient": c}))
            .collect();
        let structural_note = match structural {
            Some(sr) if rank >= sr => {
                "The structural rank shows the same loss, so the sparsity pattern alone forces it."
            }
            Some(_) => {
                "The structural rank does not show this loss, so it comes from the values at this point."
            }
            None => "",
        };
        findings.push(
            Finding::new("numerical_rank", "dependent_equality_rows", Severity::Warning,
                Basis::Numerical, Confidence::High,
                format!("At {}, equalities {} are linearly dependent, and removing any one of them leaves the rest independent (a minimal dependent set). {structural_note}",
                    pt.label, names.join(", ")))
            .why("LICQ fails at this point: the multipliers of these rows are not unique, and the interior-point method must regularize. If the rows' right-hand sides are not consistent with the dependency, no point satisfies them all.")
            .next("Check whether one of these equations is implied by the others; the combination in the evidence shows how they cancel.")
            .cons(names)
            .evidence(json!({"point": pt.label, "left_null_combination": combo,
                "relative_residual": num(set.residual), "threshold": num(tau),
                "method": "dense SVD, deletion filter at a fixed singular-value threshold"})),
        );
    }
    if let Some(set) = &search.too_large {
        let names: Vec<String> = set.iter().map(|&p| cname(prob, d.rows[p])).collect();
        findings.push(
            Finding::new("numerical_rank", "dependent_equality_rows_unlocalized", Severity::Warning,
                Basis::Numerical, Confidence::Medium,
                format!("At {}, the equality Jacobian is rank deficient by {deficiency}; {} rows take part in the null space, more than the minimal-set search limit (--max-dependent-search {}). Rows: {}.",
                    pt.label, set.len(), opts.max_dependent_search, list(&names)))
            .why("LICQ fails at this point; the solver needs dual regularization and the multipliers are not unique.")
            .next("Raise --max-dependent-search, or inspect the listed rows for a redundant equation.")
            .cons(names),
        );
    }
    let found = search.sets.len();
    if deficiency > found && found == opts.max_dependent_sets {
        findings.push(
            Finding::new("numerical_rank", "more_dependent_sets_not_searched", Severity::Info,
                Basis::Numerical, Confidence::High,
                format!("The deficiency at {} is {deficiency}; only {found} minimal dependent set(s) were searched (--max-dependent-sets).", pt.label))
            .why("The remaining dependencies are not localized in this report.")
            .next("Raise --max-dependent-sets to localize more."),
        );
    }
    let mut sink = report.ran("numerical_rank");
    for f in findings {
        sink.push(f);
    }
}

/// `numerical_rank` for an equality Jacobian too large for the dense SVD:
/// the rank from feral's inertia, without localization.
fn sparse_numerical_rank(
    prob: &NlProblem,
    pt: &PointEval,
    eq: &[usize],
    structural: Option<usize>,
    why_dense: &str,
    report: &mut Report,
) {
    let fixed = fixed_mask(prob);
    let cols: Vec<usize> = (0..prob.n).filter(|&j| !fixed[j]).collect();
    let Some(entries) = sparse_block(pt, prob.m, eq, None, &cols, prob.n) else {
        report.unavailable(
            "numerical_rank",
            format!("the Jacobian has non-finite entries at {}", pt.label),
        );
        return;
    };
    let rank = match feral_rank(eq.len(), cols.len(), &entries) {
        Ok(r) => r,
        Err(e) => {
            report.unavailable(
                "numerical_rank",
                format!("{why_dense}; and the sparse inertia count failed: {e}"),
            );
            return;
        }
    };
    let m = eq.len();
    let deficiency = m - rank;
    report.sections.insert(
        "numerical_rank",
        json!({"point": pt.label, "equality_rows": m, "columns": cols.len(), "rank": rank,
            "deficiency": deficiency, "method": "inertia of [[I, Jᵀ], [J, 0]] (feral)",
            "dense_svd_skipped": why_dense, "structural_rank": structural}),
    );
    let mut sink = report.ran("numerical_rank");
    if deficiency > 0 {
        let value_note = match structural {
            Some(sr) if rank < sr => format!(
                " The structural rank is {sr}, so a deficiency of {} comes from the values, not the sparsity pattern.",
                sr - rank
            ),
            _ => String::new(),
        };
        sink.push(
            Finding::new("numerical_rank", "equality_rank_deficient_unlocalized", Severity::Warning,
                Basis::Numerical, Confidence::Medium,
                format!("At {}, the equality Jacobian ({m} × {}) has rank {rank}, a deficiency of {deficiency}, by the inertia of [[I, Jᵀ], [J, 0]].{value_note} The block is too large for the dense search that names the dependent rows.", pt.label, cols.len()))
            .why("LICQ fails at this point: the solver needs dual regularization and the multipliers are not unique.")
            .next("Raise --max-dense-entries / --max-dense-dim to localize the dependent rows, or look for duplicated balance equations.")
            .evidence(json!({"point": pt.label, "rank": rank, "deficiency": deficiency, "method": "feral inertia", "dense_svd_skipped": why_dense})),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, NlProblemParts};

    fn dense_entries(rows: &[&[Number]]) -> Vec<(usize, usize, Number)> {
        let mut out = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            for (j, &a) in r.iter().enumerate() {
                if a != 0.0 {
                    out.push((i, j, a));
                }
            }
        }
        out
    }

    #[test]
    fn feral_inertia_gives_the_rank() {
        let cases: [(&[&[Number]], usize, usize); 6] = [
            (&[&[1.0, 1.0], &[1.0, -1.0]], 2, 2),
            (
                &[&[1.0, 1.0, 0.0], &[0.0, 0.0, 1.0], &[2.0, 2.0, 0.0]],
                3,
                2,
            ),
            (
                &[
                    &[1.0, 1.0, 0.0, 0.0],
                    &[0.0, 1.0, 1.0, 0.0],
                    &[1.0, 0.0, -1.0, 0.0],
                    &[0.0, 0.0, 0.0, 1.0],
                ],
                4,
                3,
            ),
            (&[&[2.0, 2.0], &[1.0, 1.0]], 2, 1),
            (&[&[1.0], &[3.0], &[2.0]], 1, 1),
            // A tiny but independent row: feral scales, so it is not dependent.
            (&[&[1.0, 1.0], &[1e-20, -1e-20]], 2, 2),
        ];
        for (rows, n, want) in cases {
            assert_eq!(
                feral_rank(rows.len(), n, &dense_entries(rows)).unwrap(),
                want,
                "{rows:?}"
            );
        }
        assert_eq!(feral_rank(0, 3, &[]).unwrap(), 0);
        assert_eq!(feral_rank(2, 0, &[]).unwrap(), 0);
    }

    fn v(j: usize) -> Expr {
        Expr::Var(j)
    }
    fn c(x: Number) -> Expr {
        Expr::Const(x)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Add, Box::new(a), Box::new(b))
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn eqs(cons: Vec<Expr>, n: usize) -> NlProblem {
        let m = cons.len();
        NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: cons,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![1.0; n],
            g_l: vec![0.0; m],
            g_u: vec![0.0; m],
            var_names: Vec::new(),
            con_names: (0..m).map(|i| format!("e{i}")).collect(),
        })
        .unwrap()
    }
    fn run(p: &NlProblem, x: Vec<Number>) -> Report {
        let pt = PointEval::evaluate_nl(p, x, "test point".into());
        let mut r = Report::new("t".into(), p.n, p.m, 50);
        let sr = check_structural(p, &pt, &mut r);
        check_numerical(p, &pt, sr, &DiagnoseOptions::default(), &mut r);
        r
    }
    fn dep_sets(r: &Report) -> Vec<Vec<String>> {
        r.findings
            .iter()
            .filter(|f| f.code == "dependent_equality_rows")
            .map(|f| f.constraints.clone())
            .collect()
    }

    #[test]
    fn duplicate_rows_form_a_minimal_pair() {
        // e0: x0 + x1, e1: x2, e2: 2 x0 + 2 x1.
        let p = eqs(
            vec![
                add(v(0), v(1)),
                v(2),
                add(mul(c(2.0), v(0)), mul(c(2.0), v(1))),
            ],
            3,
        );
        let r = run(&p, vec![1.0; 3]);
        assert_eq!(dep_sets(&r), vec![vec!["e0".to_string(), "e2".to_string()]]);
    }

    #[test]
    fn three_row_dependence_with_no_dependent_pair() {
        // e0: x0 + x1, e1: x1 + x2, e2: x0 - x2 (= e0 - e1); e3: x3.
        let p = eqs(
            vec![
                add(v(0), v(1)),
                add(v(1), v(2)),
                add(v(0), mul(c(-1.0), v(2))),
                v(3),
            ],
            4,
        );
        let r = run(&p, vec![1.0; 4]);
        let sets = dep_sets(&r);
        assert_eq!(
            sets,
            vec![vec!["e0".to_string(), "e1".to_string(), "e2".to_string()]]
        );
        // Structurally fine: three rows, three columns, a perfect matching.
        assert!(
            !r.findings
                .iter()
                .any(|f| f.code == "structurally_overdetermined_equalities")
        );
    }

    #[test]
    fn value_dependent_rank_loss_is_numerical_only() {
        // e0: x0*x1, e1: x0 + x1. Jacobian [[x1, x0], [1, 1]] is singular
        // where x0 == x1, and regular elsewhere.
        let p = eqs(vec![mul(v(0), v(1)), add(v(0), v(1))], 2);
        let r = run(&p, vec![2.0, 2.0]);
        let codes: Vec<_> = r.findings.iter().map(|f| f.code).collect();
        assert!(
            codes.contains(&"numerical_rank_below_structural_rank"),
            "{codes:?}"
        );
        assert_eq!(dep_sets(&r), vec![vec!["e0".to_string(), "e1".to_string()]]);
        let r = run(&p, vec![1.0, 3.0]);
        assert!(
            r.findings.is_empty(),
            "regular point: {:?}",
            r.findings.iter().map(|f| f.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn structural_overdetermination() {
        // Two equations in the single variable x0.
        let p = eqs(vec![v(0), mul(c(3.0), v(0))], 1);
        let r = run(&p, vec![0.0]);
        let codes: Vec<_> = r.findings.iter().map(|f| f.code).collect();
        assert!(
            codes.contains(&"structurally_overdetermined_equalities"),
            "{codes:?}"
        );
        assert!(!codes.contains(&"numerical_rank_below_structural_rank"));
        assert_eq!(dep_sets(&r).len(), 1);
    }

    #[test]
    fn full_rank_reports_nothing() {
        let p = eqs(vec![add(v(0), v(1)), add(v(0), mul(c(-1.0), v(1)))], 2);
        assert!(run(&p, vec![1.0, 1.0]).findings.is_empty());
    }
}
