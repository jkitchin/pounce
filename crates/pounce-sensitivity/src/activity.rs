//! Post-solve activity classification (the covariance/information
//! roadmap's item 0, gh #362).
//!
//! Classifies every bounded variable and every finite-bounded inequality
//! row of a converged barrier solve into one of five statuses, keyed on
//! the ratio of barrier curvature to the model's own curvature:
//!
//! ```text
//! r = Σ / q,   Σ = z/s summed over the sides that exist,
//!              q = |H_ii|                        (variable)
//!                  |∇dⱼᵀ H ∇dⱼ| / ‖∇dⱼ‖⁴         (inequality row)
//! ```
//!
//! The row denominator carries the fourth power so that `r` is
//! invariant to rescaling the row: `d → c·d` sends `Σ → Σ/c²` while
//! the curvature along the unit normal is unchanged, and `‖∇d‖⁴`
//! restores the balance. Equivalently, the geometric barrier weight
//! `Σ‖∇d‖²` (distance to the surface is `d/‖∇d‖`, its conjugate
//! multiplier `v‖∇d‖`) is measured against the curvature along the
//! unit normal. Variable bounds are invariant as written. This also
//! absorbs the solver's own per-row `d_scale`.
//!
//! `H` is the exact Lagrangian Hessian, so constraint curvature
//! contributes to `q` alongside the objective's. For variables, `q`
//! reads the Hessian DIAGONAL only, so purely off-diagonal coupling is
//! invisible to it: `f = x₁x₂` with bounds on both variables reports
//! `unidentified` on every bound even though the bound directions have
//! well-defined curvature. Items 1-4 of the covariance roadmap inherit
//! these semantics where they consume the per-coordinate statuses;
//! their reduced-block classification is where coupling becomes
//! visible, folded into the reduced diagonal by elimination.
//!
//! `r` is `O(μ)` when the bound is inactive, `O(1)` when weakly active
//! (slack and multiplier vanish together), and `O(1/μ)` when strongly
//! active, so one ratio separates the regimes at any `μ` where a fixed
//! threshold on the slack or the multiplier alone cannot: both are
//! `O(√μ)` at weak activity, so any constant tracks the solve rather
//! than the geometry.
//!
//! Everything read here is retained by the converged state the
//! backsolver already holds: the bound multipliers on the iterate, the
//! solver's own slacks, `Σ` as `curr_sigma_x` / `curr_sigma_s`, the
//! barrier parameter, and the exact Lagrangian Hessian, so `H` is
//! never recovered from the barrier-augmented factor.
//!
//! The report is indexed in **user space**: `var_*` arrays have the
//! user TNLP's full variable count and `row_*` arrays its full
//! constraint count. A variable removed internally by
//! `fixed_variable_treatment = make_parameter` (`lb == ub`, the
//! default) reports [`FIXED`] at its own user index, and an equality
//! constraint reports [`EQUALITY`], so user indices never shift.

use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_linalg::Matrix;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use pounce_linalg::triplet::{GenTMatrix, SymTMatrix};

use crate::PdSensBacksolver;
use crate::vec_util::dense_to_vec;

/// No finite bound on this variable or row: nothing to classify.
pub const UNBOUNDED: i8 = -1;
/// `r = O(μ)`: the bound is not doing anything.
pub const INACTIVE: i8 = 0;
/// `r = O(1)`: slack and multiplier vanish together; kept, flagged.
pub const WEAKLY_ACTIVE: i8 = 1;
/// `r = O(1/μ)`: the bound holds the variable; projected out.
pub const STRONGLY_ACTIVE: i8 = 2;
/// `r` in a gap between the band and a `μ`-edge: undetermined at this
/// `μ`; re-solving tighter separates it.
pub const AMBIGUOUS: i8 = 3;
/// The curvature `q` is below noise scale: the bound question does not
/// arise, and the direction is poorly identified.
pub const UNIDENTIFIED: i8 = 4;
/// `lb == ub`: the variable was removed from the solve as a parameter
/// (`fixed_variable_treatment = make_parameter`), so there is no
/// barrier geometry to classify.
pub const FIXED: i8 = 5;
/// An equality constraint: always active by construction, with no
/// slack or multiplier pair on the barrier, so outside this
/// classification.
pub const EQUALITY: i8 = 6;

/// Per-variable and per-row classification of a converged solve.
///
/// All vectors are **user-space**: `var_*` have length `n_full_x` (the
/// user TNLP's `n`) and `row_*` length `n_full_g` (the user's `m`).
/// Entries with no finite bound hold [`UNBOUNDED`]; [`FIXED`]
/// variables and [`EQUALITY`] rows are placeholders for entries the
/// barrier never classified. All three carry `NaN` ratios.
pub struct ActivityReport {
    /// Barrier parameter of the converged iterate.
    pub mu: Number,
    /// Status per user variable (codes above).
    pub var_status: Vec<i8>,
    /// `Σ_i / q_i` per user variable; `NaN` where not classified.
    /// For an [`UNIDENTIFIED`] entry the value is `Σ/floor`, a lower
    /// bound on any honest ratio rather than the ratio itself, since
    /// `q` is below the identification floor there.
    pub var_ratio: Vec<Number>,
    /// Sign of the signed curvature `H_ii` (−1, 0, +1); the absolute
    /// value goes into `q`, so an indefinite direction is reported
    /// rather than hidden.
    pub var_q_sign: Vec<i8>,
    /// `s·z` differs from `μ` by more than a factor of ten on some
    /// side: off the central path, or the bound was relaxed.
    pub var_off_central_path: Vec<bool>,
    /// Classified inactive yet `r` non-negligible: barrier curvature
    /// where none should be.
    pub var_contaminated: Vec<bool>,
    /// The barrier diagonal `Σ_i = z/s` itself per user variable, both
    /// sides summed; 0 where not classified. In **natural (unscaled)
    /// units**, the repo's sensitivity-output contract: classification
    /// runs on the solver's scaled quantities (the ratio is
    /// scale-invariant), the report does not. The covariance roadmap's
    /// item 1 subtracts exactly this from the factor's natural-units
    /// reduced Hessian.
    pub var_sigma: Vec<Number>,
    /// Status per user constraint row.
    pub row_status: Vec<i8>,
    /// `Σ_j / q_j` per user row; `NaN` where not classified.
    /// [`UNIDENTIFIED`] entries hold `Σ/floor` as for variables.
    pub row_ratio: Vec<Number>,
    /// Sign of the signed row curvature `∇dⱼᵀ H ∇dⱼ`.
    pub row_q_sign: Vec<i8>,
    /// Central-path check per row, as for variables.
    pub row_off_central_path: Vec<bool>,
    /// Contamination check per row, as for variables.
    pub row_contaminated: Vec<bool>,
    /// The row barrier diagonal `Σ_j = v/s` per user row, both sides
    /// summed; 0 where not classified. In **natural (unscaled) units**
    /// like [`Self::var_sigma`], and RAW rather than the geometric
    /// weight the classification uses: item 1 restricts the normal to
    /// its own fitted block and applies its own `‖a‖²` factor there.
    pub row_sigma: Vec<Number>,
}

/// The classification rule of the roadmap's item 0.
fn classify(r: Number, mu: Number) -> i8 {
    if mu > 1e-4 {
        // The band is fixed at [1e-1, 1e1] while the μ-edges √μ and
        // 1/√μ move with the solve: they meet the band at μ = 1e-2,
        // and a full decade separates them from it at μ = 1e-4. Above
        // 1e-4 that margin is what's thinning, so only the two calls
        // that stay clear are made and the middle is honest refusal.
        if r < 1e-1 {
            INACTIVE
        } else if r > 1e1 {
            STRONGLY_ACTIVE
        } else {
            AMBIGUOUS
        }
    } else if r < mu.sqrt() {
        INACTIVE
    } else if r > 1.0 / mu.sqrt() {
        STRONGLY_ACTIVE
    } else if (1e-1..=1e1).contains(&r) {
        WEAKLY_ACTIVE
    } else {
        AMBIGUOUS
    }
}

fn sign_of(x: Number) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Scatter a compressed (bounded-entries-only) vector to full length
/// through its expansion matrix. Entries without that bound stay 0.
fn expand(compressed: &[Number], px: &Rc<dyn Matrix>, n: usize) -> Vec<Number> {
    let em = px
        .as_any()
        .downcast_ref::<ExpansionMatrix>()
        .expect("bound projection is an ExpansionMatrix (orig_ipopt_nlp builds no other kind)");
    let idx = em.expanded_pos_indices();
    assert_eq!(
        idx.len(),
        compressed.len(),
        "compressed bound vector length disagrees with its expansion",
    );
    let mut full = vec![0.0; n];
    for (k, &pos) in idx.iter().enumerate() {
        full[pos as usize] = compressed[k];
    }
    full
}

/// Presence mask for a bound side, from the same expansion.
fn present(px: &Rc<dyn Matrix>, n: usize) -> Vec<bool> {
    let em = px
        .as_any()
        .downcast_ref::<ExpansionMatrix>()
        .expect("bound projection is an ExpansionMatrix (orig_ipopt_nlp builds no other kind)");
    let mut mask = vec![false; n];
    for &pos in em.expanded_pos_indices() {
        mask[pos as usize] = true;
    }
    mask
}

/// The exact Hessian diagonal: one pass over the triplet structure for
/// the type `eval_h` builds today. The mat-vec fallback keeps any
/// future non-triplet `SymMatrix` correct, at O(n·nnz) cost.
fn hessian_diagonal(hess: &Rc<dyn pounce_linalg::SymMatrix>, n: usize) -> Vec<Number> {
    let mut diag = vec![0.0; n];
    if let Some(t) = hess.as_any().downcast_ref::<SymTMatrix>() {
        // triplet indices are 1-based (the GenTMatrix convention);
        // duplicates accumulate, matching mult_vector
        for ((&i, &j), &v) in t.irows().iter().zip(t.jcols()).zip(t.values()) {
            if i == j {
                diag[(i - 1) as usize] += v;
            }
        }
        return diag;
    }
    let space = DenseVectorSpace::new(n as i32);
    let mut e = DenseVector::new(space.clone());
    let mut he = DenseVector::new(space);
    for (i, d) in diag.iter_mut().enumerate() {
        e.values_mut().fill(0.0);
        e.values_mut()[i] = 1.0;
        he.values_mut().fill(0.0);
        hess.mult_vector(1.0, &e, 0.0, &mut he);
        // values_mut, not values: a zero product may have left the
        // output homogeneous (empty backing slice); this materializes
        *d = he.values_mut()[i];
    }
    diag
}

/// A bounded row whose gradient vanishes at the iterate has no
/// direction to measure curvature along: unidentified, exactly as a
/// below-floor `q`, never `unbounded` (the bounds are real). The
/// ratio is the raw `Σ/floor` lower bound; the geometric weight is
/// degenerate at zero gradient.
fn zero_gradient_row(sigma: Number, floor: Number) -> Entry {
    Entry {
        status: UNIDENTIFIED,
        ratio: sigma / floor,
        q_sign: 0,
        off_path: false,
        contaminated: false,
        sigma,
    }
}

/// Central-path check for one side: `s·z` within a factor of ten of `μ`.
fn off_path(s: Number, z: Number, mu: Number) -> bool {
    let comp = s * z;
    comp > 10.0 * mu || comp < 0.1 * mu
}

/// Classified inactive yet `r` well above the `O(μ)` an inactive
/// bound should carry: barrier curvature where none should be. The
/// threshold is μ-relative because `inactive` MEANS `r = O(μ)`; a
/// fixed constant can never sit below the inactive edge `√μ` at any
/// converged μ (second review).
fn contaminated(status: i8, r: Number, mu: Number) -> bool {
    status == INACTIVE && r > 100.0 * mu
}

/// One classified entry in internal space, before the user-space
/// scatter.
#[derive(Clone, Copy)]
struct Entry {
    status: i8,
    ratio: Number,
    q_sign: i8,
    off_path: bool,
    contaminated: bool,
    /// The RAW barrier diagonal, whatever weight classification used.
    sigma: Number,
}

const NOT_CLASSIFIED: Entry = Entry {
    status: UNBOUNDED,
    ratio: Number::NAN,
    q_sign: 0,
    off_path: false,
    contaminated: false,
    sigma: 0.0,
};

/// Classify one bounded variable or row from its `Σ` and signed `q`.
/// `off_path` is the caller's to fill: it reads the per-side slack and
/// multiplier, not the ratio.
fn classify_entry(sigma: Number, q_signed: Number, floor: Number, mu: Number) -> Entry {
    let q_sign = sign_of(q_signed);
    let q = q_signed.abs();
    if q < floor {
        return Entry {
            status: UNIDENTIFIED,
            ratio: sigma / floor,
            q_sign,
            off_path: false,
            contaminated: false,
            sigma,
        };
    }
    let r = sigma / q;
    let status = classify(r, mu);
    Entry {
        status,
        ratio: r,
        q_sign,
        off_path: false,
        contaminated: contaminated(status, r, mu),
        sigma,
    }
}

pub(crate) fn compute(bs: &PdSensBacksolver) -> ActivityReport {
    let (data, cq, nlp) = bs.activity_handles();

    // scoped borrows: the Cq getters below re-borrow the NLP (mutably,
    // for lazy evaluation) and the data, so nothing here may hold
    // either across a Cq call
    let (mu, mult_z_l, mult_z_u, mult_v_l, mult_v_u, n, m_d) = {
        let d = data.borrow();
        let curr = d.curr.as_ref().expect("converged state has an iterate");
        (
            d.curr_mu,
            Rc::clone(&curr.z_l),
            Rc::clone(&curr.z_u),
            Rc::clone(&curr.v_l),
            Rc::clone(&curr.v_u),
            curr.x.dim() as usize,
            curr.s.dim() as usize,
        )
    };
    let (px_l, px_u, pd_l, pd_u, obj_scale, d_scale) = {
        let nl = nlp.borrow();
        (
            nl.px_l(),
            nl.px_u(),
            nl.pd_l(),
            nl.pd_u(),
            nl.obj_scaling_factor(),
            nl.d_scale_vec(),
        )
    };
    let cq = cq.borrow();

    // --- variables, in internal space ------------------------------------
    let has_l = present(&px_l, n);
    let has_u = present(&px_u, n);
    let z_l = expand(&dense_to_vec(mult_z_l.as_ref()), &px_l, n);
    let z_u = expand(&dense_to_vec(mult_z_u.as_ref()), &px_u, n);
    let s_l = expand(&dense_to_vec(cq.curr_slack_x_l().as_ref()), &px_l, n);
    let s_u = expand(&dense_to_vec(cq.curr_slack_x_u().as_ref()), &px_u, n);
    let sigma_x = dense_to_vec(cq.curr_sigma_x().as_ref());

    let hess = cq.curr_exact_hessian();
    // the identification floor is relative to the largest curvature
    // anywhere on the diagonal, not just the bounded entries, so a
    // row-only model still measures q against the model's own scale
    let diag = hessian_diagonal(&hess, n);
    let max_abs_diag = diag.iter().fold(0.0, |a: Number, d| a.max(d.abs()));
    let floor = Number::EPSILON.sqrt() * max_abs_diag.max(1.0);

    let mut vars = vec![NOT_CLASSIFIED; n];
    for i in 0..n {
        if !(has_l[i] || has_u[i]) {
            continue;
        }
        let mut e = classify_entry(sigma_x[i], diag[i], floor, mu);
        e.off_path = (has_l[i] && off_path(s_l[i], z_l[i], mu))
            || (has_u[i] && off_path(s_u[i], z_u[i], mu));
        // the ratio is scale-invariant, so classification ran in the
        // solver's scaled space; the REPORTED sigma follows the repo's
        // natural-units contract: internal z carries the objective
        // scale (x is never scaled), so Sigma_nat = Sigma / df
        e.sigma /= obj_scale;
        vars[i] = e;
    }

    // --- inequality rows, in internal space -------------------------------
    let rhas_l = present(&pd_l, m_d);
    let rhas_u = present(&pd_u, m_d);
    let v_l = expand(&dense_to_vec(mult_v_l.as_ref()), &pd_l, m_d);
    let v_u = expand(&dense_to_vec(mult_v_u.as_ref()), &pd_u, m_d);
    let rs_l = expand(&dense_to_vec(cq.curr_slack_s_l().as_ref()), &pd_l, m_d);
    let rs_u = expand(&dense_to_vec(cq.curr_slack_s_u().as_ref()), &pd_u, m_d);
    let sigma_s = dense_to_vec(cq.curr_sigma_s().as_ref());

    let jac_d = cq.curr_jac_d();
    // One pass over the Jacobian triplets gathers every row's support
    // and one pass over the Hessian triplets builds an adjacency view,
    // so each row's curvature costs its own support times its
    // neighbours instead of a full mat-vec pair per row (second
    // review). The mat-vec loop below remains the fallback for any
    // future non-triplet matrix types.
    let mut rows = vec![NOT_CLASSIFIED; m_d];
    let fast = match (
        jac_d.as_any().downcast_ref::<GenTMatrix>(),
        hess.as_any().downcast_ref::<SymTMatrix>(),
    ) {
        (Some(jt), Some(ht)) => {
            // gather and merge each row's entries (triplet duplicates
            // sum, matching mult_vector; indices are 1-based)
            let mut support: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m_d];
            for ((&r, &c), &v) in jt.irows().iter().zip(jt.jcols()).zip(jt.values()) {
                support[(r - 1) as usize].push(((c - 1) as usize, v));
            }
            for sup in &mut support {
                sup.sort_unstable_by_key(|&(c, _)| c);
                sup.dedup_by(|a, b| {
                    if a.0 == b.0 {
                        b.1 += a.1;
                        true
                    } else {
                        false
                    }
                });
            }
            let mut adj: Vec<Vec<(usize, Number)>> = vec![Vec::new(); n];
            for ((&i, &l), &v) in ht.irows().iter().zip(ht.jcols()).zip(ht.values()) {
                let (a, b) = ((i - 1) as usize, (l - 1) as usize);
                adj[a].push((b, v));
                if a != b {
                    adj[b].push((a, v));
                }
            }
            let mut scratch = vec![0.0; n];
            for j in 0..m_d {
                if !(rhas_l[j] || rhas_u[j]) {
                    continue;
                }
                let sup = &support[j];
                let norm2: Number = sup.iter().map(|&(_, g)| g * g).sum();
                rows[j] = if norm2 <= 0.0 {
                    zero_gradient_row(sigma_s[j], floor)
                } else {
                    for &(k, g) in sup {
                        scratch[k] = g;
                    }
                    let mut ghg = 0.0;
                    for &(k, gk) in sup {
                        let mut acc = 0.0;
                        for &(l, v) in &adj[k] {
                            acc += v * scratch[l];
                        }
                        ghg += gk * acc;
                    }
                    for &(k, _) in sup {
                        scratch[k] = 0.0;
                    }
                    // Σ·‖∇d‖² against curvature along the unit
                    // normal: invariant to rescaling the row; the
                    // report keeps the raw Σ
                    let mut e = classify_entry(sigma_s[j] * norm2, ghg / norm2, floor, mu);
                    e.sigma = sigma_s[j];
                    e
                };
            }
            true
        }
        _ => false,
    };
    if !fast {
        let mspace = DenseVectorSpace::new(m_d as i32);
        let mut e_row = DenseVector::new(mspace);
        let nspace = DenseVectorSpace::new(n as i32);
        let mut grad = DenseVector::new(nspace.clone());
        let mut hgrad = DenseVector::new(nspace);
        for j in 0..m_d {
            if !(rhas_l[j] || rhas_u[j]) {
                continue;
            }
            // ∇dⱼ = Jdᵀ eⱼ, then the curvature along the normal;
            // values_mut throughout because a zero product may leave
            // the output homogeneous (empty backing slice)
            e_row.values_mut().fill(0.0);
            e_row.values_mut()[j] = 1.0;
            grad.values_mut().fill(0.0);
            jac_d.trans_mult_vector(1.0, &e_row, 0.0, &mut grad);
            let norm2: Number = grad.values_mut().iter().map(|g| *g * *g).sum();
            rows[j] = if norm2 <= 0.0 {
                zero_gradient_row(sigma_s[j], floor)
            } else {
                hgrad.values_mut().fill(0.0);
                hess.mult_vector(1.0, &grad, 0.0, &mut hgrad);
                let ghg: Number = {
                    let h = hgrad.values_mut();
                    grad.values_mut()
                        .iter()
                        .zip(h.iter())
                        .map(|(g, h)| g * h)
                        .sum()
                };
                let mut e = classify_entry(sigma_s[j] * norm2, ghg / norm2, floor, mu);
                e.sigma = sigma_s[j];
                e
            };
        }
    }
    for j in 0..m_d {
        if !(rhas_l[j] || rhas_u[j]) {
            continue;
        }
        rows[j].off_path = (rhas_l[j] && off_path(rs_l[j], v_l[j], mu))
            || (rhas_u[j] && off_path(rs_u[j], v_u[j], mu));
        // natural-units report, as for variables: the scaled row
        // multiplier carries df/dg and the scaled slack dg, so
        // Sigma_nat = Sigma * dg^2 / df
        let dg = d_scale.as_ref().map_or(1.0, |v| v[j]);
        rows[j].sigma *= dg * dg / obj_scale;
    }

    // --- scatter to user space --------------------------------------------
    // all Cq evaluation is done, so borrowing the NLP again is safe
    let nl = nlp.borrow();
    let n_full_x = nl.n_full_x() as usize;
    let n_full_g = nl.n_full_g() as usize;

    let fixed_entry = Entry {
        status: FIXED,
        ..NOT_CLASSIFIED
    };
    let mut var_full = vec![fixed_entry; n_full_x];
    for (i, e) in vars.iter().enumerate() {
        var_full[nl.var_x_to_full_x(i as Index) as usize] = *e;
    }

    let equality_entry = Entry {
        status: EQUALITY,
        ..NOT_CLASSIFIED
    };
    let mut row_full = vec![equality_entry; n_full_g];
    // BoundClassification's d_map is one ascending scan over the
    // user's g, so the j-th full-g index outside the c-block is
    // internal inequality row j
    let mut d_pos = 0usize;
    for (full_idx, slot) in row_full.iter_mut().enumerate() {
        if nl.full_g_to_c_block(full_idx as Index).is_none() {
            *slot = rows[d_pos];
            d_pos += 1;
        }
    }
    assert_eq!(d_pos, m_d, "inequality count disagrees with the c/d split");

    ActivityReport {
        mu,
        var_status: var_full.iter().map(|e| e.status).collect(),
        var_ratio: var_full.iter().map(|e| e.ratio).collect(),
        var_q_sign: var_full.iter().map(|e| e.q_sign).collect(),
        var_off_central_path: var_full.iter().map(|e| e.off_path).collect(),
        var_contaminated: var_full.iter().map(|e| e.contaminated).collect(),
        var_sigma: var_full.iter().map(|e| e.sigma).collect(),
        row_status: row_full.iter().map(|e| e.status).collect(),
        row_ratio: row_full.iter().map(|e| e.ratio).collect(),
        row_q_sign: row_full.iter().map(|e| e.q_sign).collect(),
        row_off_central_path: row_full.iter().map(|e| e.off_path).collect(),
        row_contaminated: row_full.iter().map(|e| e.contaminated).collect(),
        row_sigma: row_full.iter().map(|e| e.sigma).collect(),
    }
}

/// The gradient of one user constraint row at the converged iterate,
/// in user variable order (length `n_full_x`) and **natural (unscaled)
/// units**: the internal Jacobian row carries the solver's per-row
/// scale, which is divided out here per the sensitivity-output
/// contract. Works for equality and inequality rows alike; entries for
/// `make_parameter`-removed fixed variables are 0 because the solve
/// dropped their columns.
pub(crate) fn row_normal(bs: &PdSensBacksolver, user_row: usize) -> Result<Vec<Number>, usize> {
    let (data, cq, nlp) = bs.activity_handles();
    let n = {
        let d = data.borrow();
        d.curr
            .as_ref()
            .expect("converged state has an iterate")
            .x
            .dim() as usize
    };
    // position of the row within its own c/d block, by the same
    // ascending scan the report's scatter uses
    let c_pos = {
        let nl = nlp.borrow();
        if user_row >= nl.n_full_g() as usize {
            return Err(nl.n_full_g() as usize);
        }
        nl.full_g_to_c_block(user_row as Index)
    };
    let block_pos = match c_pos {
        Some(p) => p as usize,
        None => {
            let nl = nlp.borrow();
            (0..user_row)
                .filter(|&g| nl.full_g_to_c_block(g as Index).is_none())
                .count()
        }
    };

    let row_scale = {
        let nl = nlp.borrow();
        let sv = if c_pos.is_some() {
            nl.c_scale_vec()
        } else {
            nl.d_scale_vec()
        };
        sv.map_or(1.0, |v| v[block_pos])
    };
    let cq = cq.borrow();
    let jac = if c_pos.is_some() {
        cq.curr_jac_c()
    } else {
        cq.curr_jac_d()
    };
    let m_block = jac.n_rows() as usize;
    let mspace = DenseVectorSpace::new(m_block as i32);
    let mut e_row = DenseVector::new(mspace);
    let nspace = DenseVectorSpace::new(n as i32);
    let mut grad = DenseVector::new(nspace);
    e_row.values_mut().fill(0.0);
    e_row.values_mut()[block_pos] = 1.0;
    grad.values_mut().fill(0.0);
    jac.trans_mult_vector(1.0, &e_row, 0.0, &mut grad);

    let nl = nlp.borrow();
    let n_full_x = nl.n_full_x() as usize;
    let mut full = vec![0.0; n_full_x];
    let g = grad.values_mut();
    for (i, slot) in g.iter().enumerate() {
        full[nl.var_x_to_full_x(i as Index) as usize] = *slot / row_scale;
    }
    Ok(full)
}

/// The exact Lagrangian Hessian times a user-space vector, in user
/// variable order and **natural (unscaled) units**: the internal
/// Hessian carries the objective scale, divided out here per the
/// sensitivity-output contract. Entries for `make_parameter`-removed
/// fixed variables are 0 in and out (their columns left the solve).
/// Serves the covariance roadmap's item 2: the tangent-recovered
/// reduced Hessian is `T^T (H T)`, one product per fitted column.
pub(crate) fn hessian_vec(bs: &PdSensBacksolver, v_full: &[Number]) -> Result<Vec<Number>, usize> {
    let (data, cq, nlp) = bs.activity_handles();
    let n = {
        let d = data.borrow();
        d.curr
            .as_ref()
            .expect("converged state has an iterate")
            .x
            .dim() as usize
    };
    let (n_full_x, obj_scale) = {
        let nl = nlp.borrow();
        (nl.n_full_x() as usize, nl.obj_scaling_factor())
    };
    if v_full.len() != n_full_x {
        return Err(n_full_x);
    }

    let nspace = DenseVectorSpace::new(n as i32);
    let mut v_int = DenseVector::new(nspace.clone());
    let mut hv = DenseVector::new(nspace);
    {
        let nl = nlp.borrow();
        let vals = v_int.values_mut();
        vals.fill(0.0);
        for i in 0..n {
            vals[i] = v_full[nl.var_x_to_full_x(i as Index) as usize];
        }
    }
    let hess = {
        let cq = cq.borrow();
        cq.curr_exact_hessian()
    };
    hess.mult_vector(1.0, &v_int, 0.0, &mut hv);

    let nl = nlp.borrow();
    let mut out = vec![0.0; n_full_x];
    let h = hv.values_mut();
    for (i, slot) in h.iter().enumerate() {
        out[nl.var_x_to_full_x(i as Index) as usize] = *slot / obj_scale;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tight_mu_walks_all_five_regions() {
        let mu = 1e-10; // edges at 1e-5 and 1e5
        assert_eq!(classify(0.9e-5, mu), INACTIVE);
        assert_eq!(classify(1.1e-5, mu), AMBIGUOUS); // gap: edge..band
        assert_eq!(classify(0.5, mu), WEAKLY_ACTIVE);
        assert_eq!(classify(50.0, mu), AMBIGUOUS); // gap: band..edge
        assert_eq!(classify(2e5, mu), STRONGLY_ACTIVE);
    }

    #[test]
    fn band_edges_are_inclusive_and_mu_edges_separate() {
        let mu = 1e-10;
        assert_eq!(classify(1e-1, mu), WEAKLY_ACTIVE);
        assert_eq!(classify(1e1, mu), WEAKLY_ACTIVE);
        // either side of each μ-edge (exactly-on is float-fragile:
        // √(1e-10) is not exactly 1e-5)
        assert_eq!(classify(0.99e-5, mu), INACTIVE);
        assert_eq!(classify(1.01e-5, mu), AMBIGUOUS);
        assert_eq!(classify(0.99e5, mu), AMBIGUOUS);
        assert_eq!(classify(1.01e5, mu), STRONGLY_ACTIVE);
    }

    #[test]
    fn loose_mu_refuses_the_weak_call() {
        // μ > 1e-4: three statuses only, the band reports ambiguous
        for mu in [1e-3, 1e-2, 1e-1] {
            assert_eq!(classify(0.05, mu), INACTIVE);
            assert_eq!(classify(1.0, mu), AMBIGUOUS);
            assert_eq!(classify(50.0, mu), STRONGLY_ACTIVE);
        }
        // at μ = 1e-4 exactly the μ-branch is not taken: the weak call
        // is available, with a decade of margin edge-to-band
        assert_eq!(classify(1.0, 1e-4), WEAKLY_ACTIVE);
    }

    #[test]
    fn off_path_is_a_factor_of_ten_both_ways() {
        let mu = 1e-2;
        assert!(!off_path(1.0, 1e-2, mu)); // s·z = μ exactly
        assert!(!off_path(0.5, 1e-2, mu)); // within 10×
        assert!(off_path(1.0, 0.2, mu)); // 20× above
        assert!(off_path(1.0, 5e-4, mu)); // 20× below
    }

    #[test]
    fn contamination_is_mu_relative_and_inactive_only() {
        let mu = 1e-10; // inactive edge at 1e-5, threshold at 1e-8
        assert!(contaminated(INACTIVE, 1e-6, mu));
        assert!(!contaminated(INACTIVE, 5e-9, mu));
        assert!(!contaminated(WEAKLY_ACTIVE, 1.0, mu));
        assert!(!contaminated(STRONGLY_ACTIVE, 1e5, mu));
        // the flag is reachable: 100μ sits below the inactive edge √μ
        // whenever μ < 1e-4, so an inactive r can exceed it
        assert!(100.0 * mu < mu.sqrt());
    }

    #[test]
    fn below_floor_reports_unidentified_with_the_sign() {
        let e = classify_entry(0.5, 1e-12, 1e-8, 1e-10);
        assert_eq!(e.status, UNIDENTIFIED);
        assert_eq!(e.q_sign, 1);
        let e = classify_entry(0.5, -1e-12, 1e-8, 1e-10);
        assert_eq!(e.status, UNIDENTIFIED);
        assert_eq!(e.q_sign, -1);
        // negative curvature above the floor classifies on |q| but
        // keeps its sign visible
        let e = classify_entry(1.0, -2.0, 1e-8, 1e-10);
        assert_eq!(e.status, WEAKLY_ACTIVE);
        assert_eq!(e.q_sign, -1);
        assert_eq!(e.ratio, 0.5);
    }
}
