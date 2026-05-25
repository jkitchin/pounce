//! Parametric-sensitivity and reduced-Hessian post-processing for the
//! `pounce` driver.
//!
//! This is the suffix-driven sIPOPT path: when an AMPL `.nl` declares
//! the sIPOPT-style suffixes (`sens_state_1`, `sens_state_value_1`,
//! `sens_init_constr`), `pounce` runs a normal solve and then performs
//! the post-optimal sensitivity step via `pounce-sensitivity`, writing
//! the perturbed primal back into the `.sol` as a `sens_sol_state_1`
//! suffix. The `--compute-red-hessian` flag additionally computes the
//! reduced Hessian over the variables tagged by the `red_hessian`
//! integer var-suffix.
//!
//! Mirror of upstream sIPOPT's `ipopt_sens` AMPL binary
//! ([`ref/Ipopt/contrib/sIPOPT/src/AmplTNLP.cpp` etc.](../../../ref/Ipopt/contrib/sIPOPT/)),
//! limited to the metadata-measurement path that the
//! `parametric_ampl` example exercises.
//!
//! The required suffixes (otherwise the solve is a plain nominal solve):
//!
//! * `sens_state_1` — integer var-suffix tagging each parameter
//!   (value = 1..n_params, 0 for non-parameters).
//! * `sens_state_value_1` — real var-suffix carrying the perturbed
//!   parameter values.
//! * `sens_init_constr` — integer con-suffix tagging which
//!   constraint pins each parameter to its nominal value (value =
//!   1..n_params, 0 otherwise).
//!
//! See [`ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp`](../../../ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/parametricTNLP.cpp)
//! `get_var_con_metadata` for the canonical suffix shape upstream
//! emits, and pounce#16's `parametric_cpp.rs` for an end-to-end
//! cross-check against upstream's golden output.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_sensitivity::{
    IndexSchurData, PdSensBacksolver, SchurData, SensApplication, SensBacksolver, SensOptions,
};

use crate::nl_reader::NlSuffixes;
use crate::nl_writer::{SolSuffix, SolSuffixTarget, SolSuffixValues};
use crate::solve_report::SolutionSuffix;

/// True when the `.nl` declares the three sIPOPT-style suffixes that
/// drive the parametric sensitivity step. When this returns `false`,
/// `pounce` runs a plain nominal solve.
pub fn is_sensitivity_input(suffixes: &NlSuffixes) -> bool {
    suffixes.var_int.contains_key("sens_state_1")
        && suffixes.var_real.contains_key("sens_state_value_1")
        && suffixes.con_int.contains_key("sens_init_constr")
}

/// Outputs of [`try_compute_red_hessian`]: the column-major `n × n`
/// reduced Hessian (`hr`), the variable indices `var_indices` that
/// label its rows/cols (so a downstream JSON consumer can map back to
/// AMPL var names), and the optional eigendecomposition.
pub struct RedHessianResult {
    /// var-x indices (algorithm-side, length `n`) that label the
    /// rows/cols of `hr`, ordered by the 1..n slot from the AMPL
    /// `red_hessian` suffix. Fixed variables are skipped (they cannot
    /// participate in the reduced Hessian).
    pub var_indices: Vec<usize>,
    /// Column-major `n × n` reduced Hessian.
    pub hr: Vec<Number>,
    /// Optional ascending eigenvalues (length `n`).
    pub eigenvalues: Option<Vec<Number>>,
    /// Optional column-major eigenvectors (length `n²`).
    pub eigenvectors: Option<Vec<Number>>,
}

/// Run the post-optimal sensitivity step and return the perturbed
/// primal lifted onto the full-x grid (length `n_full`). Returns `None`
/// (quietly) when the required suffixes are missing — the caller then
/// writes just the nominal solution.
///
/// `boundcheck_eps` enables the single-pass clamp of `x* + Δx` onto the
/// declared `[x_l, x_u]` box (mirrors sIPOPT's `sens_boundcheck`); pass
/// `None` to skip it.
#[allow(clippy::too_many_arguments)]
pub fn compute_sens_perturbed_x(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    nlp: &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    pd: Rc<RefCell<pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver>>,
    suffixes: &NlSuffixes,
    n_full: usize,
    m_full: usize,
    x_full: &[Number],
    boundcheck_eps: Option<Number>,
) -> Option<Vec<Number>> {
    let mut dx = try_compute_sens_step(data, cq, nlp, pd, suffixes, n_full, m_full, x_full)?;
    let curr = data.borrow().curr.clone()?;
    let n_x = curr.x.dim() as usize;

    if let Some(eps) = boundcheck_eps {
        // Single-pass clamp of the primal step before scattering onto
        // the full-x grid; see pounce_sensitivity::boundcheck for the
        // algorithm.
        let x_curr_compressed: Vec<Number> = curr
            .x
            .as_any()
            .downcast_ref::<DenseVector>()
            .map(|d| d.values().to_vec())
            .unwrap_or_default();
        let mut dx_primal = dx[..n_x].to_vec();
        let n_clamped = pounce_sensitivity::boundcheck::clamp_with_nlp(
            &*nlp.borrow(),
            &x_curr_compressed,
            &mut dx_primal,
            eps,
        );
        if n_clamped > 0 {
            eprintln!("pounce: --sens-boundcheck clamped {n_clamped} primal coordinate(s)");
            dx[..n_x].copy_from_slice(&dx_primal);
        }
    }

    // Scatter the compressed primal step `dx[0..n_x_var]` back onto the
    // full-x grid; fixed variables stay at their nominal values.
    let mut x_pert = x_full.to_vec();
    let nlp_ref = nlp.borrow();
    for var_idx in 0..n_x {
        let full_idx = nlp_ref.var_x_to_full_x(var_idx as Index) as usize;
        x_pert[full_idx] += dx[var_idx];
    }
    Some(x_pert)
}

/// Convert a `.sol`-shaped suffix block into the JSON report's flat
/// representation.
pub fn sol_suffix_to_report(s: &SolSuffix) -> SolutionSuffix {
    let target = match s.target {
        SolSuffixTarget::Var => "var",
        SolSuffixTarget::Con => "con",
        SolSuffixTarget::Obj => "obj",
        SolSuffixTarget::Problem => "problem",
    }
    .to_string();
    let (kind, values, int_values) = match &s.values {
        SolSuffixValues::Real(v) => ("real".to_string(), v.clone(), Vec::new()),
        SolSuffixValues::Int(v) => ("int".to_string(), Vec::new(), v.clone()),
        SolSuffixValues::ProblemReal(v) => ("real".to_string(), vec![*v], Vec::new()),
        SolSuffixValues::ProblemInt(v) => ("int".to_string(), Vec::new(), vec![*v]),
    };
    SolutionSuffix {
        name: s.name.clone(),
        target,
        kind,
        values,
        int_values,
    }
}

/// Format a reduced Hessian (and optional eigendecomp) onto stderr.
/// Matches the style of upstream sIPOPT's
/// `SensReducedHessianCalculator.cpp` `S->Print(...)` /
/// `eigenvalues->Print(...)` calls — informational, not parsed.
pub fn print_red_hessian_to_stderr(rh: &RedHessianResult) {
    let n = rh.var_indices.len();
    eprintln!("\n=== Reduced Hessian (n={n}) ===");
    eprintln!("var indices: {:?}", rh.var_indices);
    for i in 0..n {
        let mut row = String::new();
        for j in 0..n {
            // column-major: hr[i + n*j]
            row.push_str(&format!(" {:>14.6e}", rh.hr[i + n * j]));
        }
        eprintln!(" [{i:>3}]{row}");
    }
    if let Some(w) = &rh.eigenvalues {
        eprintln!("\n=== Reduced-Hessian eigenvalues (ascending) ===");
        for (k, v) in w.iter().enumerate() {
            eprintln!(" [{k:>3}] {v:>14.6e}");
        }
    }
    eprintln!();
}

/// Read the AMPL `red_hessian` integer var-suffix from `.nl`, select
/// the tagged free variables, and compute the reduced Hessian via
/// [`SensApplication::compute_reduced_hessian`] (optionally also the
/// eigendecomposition). Returns `None` (quietly) when the suffix is
/// missing or empty.
///
/// Mirrors the `compute_red_hessian=yes` branch of upstream
/// [`SensBuilder::BuildRedHessCalc`](../../../ref/Ipopt/contrib/sIPOPT/src/SensBuilder.cpp).
pub fn try_compute_red_hessian(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    nlp: &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    pd: Rc<RefCell<pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver>>,
    suffixes: &NlSuffixes,
    compute_eigen: bool,
) -> Option<RedHessianResult> {
    let red_hessian_tags = suffixes.var_int.get("red_hessian")?;
    let max_slot = red_hessian_tags.iter().copied().max().unwrap_or(0);
    if max_slot <= 0 {
        return None;
    }
    let n_slots = max_slot as usize;

    // For each slot 1..n_slots, look up the full-x index, then map to
    // the var-x index via the IpoptNlp trait. Fixed variables (no
    // var-x mapping) are skipped with a warning.
    let nlp_ref = nlp.borrow();
    let mut full_for_slot: Vec<Option<usize>> = vec![None; n_slots];
    for (full_idx, &slot) in red_hessian_tags.iter().enumerate() {
        if slot > 0 {
            let s = slot as usize;
            if s <= n_slots {
                full_for_slot[s - 1] = Some(full_idx);
            }
        }
    }
    let mut var_indices: Vec<usize> = Vec::with_capacity(n_slots);
    for (k, slot) in full_for_slot.iter().enumerate() {
        let full_idx = match slot {
            Some(i) => *i,
            None => {
                eprintln!("pounce: red_hessian slot {} has no tagged variable", k + 1);
                return None;
            }
        };
        match nlp_ref.full_x_to_var_x(full_idx as Index) {
            Some(v) => var_indices.push(v as usize),
            None => {
                eprintln!(
                    "pounce: red_hessian slot {} tags fixed variable {} (skipping)",
                    k + 1,
                    full_idx
                );
                return None;
            }
        }
    }
    drop(nlp_ref);

    // Build the row-selector SchurData over the var-x rows directly
    // (the x block starts at compound-vector index 0).
    let rows: Vec<Index> = var_indices.iter().map(|&v| v as Index).collect();
    let signs: Vec<Index> = vec![1; var_indices.len()];
    let a_data = IndexSchurData::from_parts(rows, signs).ok()?;

    let backsolver = PdSensBacksolver::new(data, cq, nlp, pd).ok()?;
    let opts = SensOptions {
        compute_red_hessian: true,
        rh_eigendecomp: compute_eigen,
        ..SensOptions::default()
    };
    let mut app = SensApplication::new(a_data, backsolver, opts);
    let n = var_indices.len();
    let mut hr = vec![0.0; n * n];
    let (eigenvalues, eigenvectors) = if compute_eigen {
        let mut w = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        if !app.compute_reduced_hessian_eigen(&mut hr, &mut w, &mut v) {
            eprintln!("pounce: reduced-Hessian eigendecomp failed");
            return None;
        }
        (Some(w), Some(v))
    } else {
        if !app.compute_reduced_hessian(&mut hr) {
            eprintln!("pounce: reduced-Hessian computation failed");
            return None;
        }
        (None, None)
    };
    let _ = cq;
    Some(RedHessianResult {
        var_indices,
        hr,
        eigenvalues,
        eigenvectors,
    })
}

/// Try to compute the parametric sensitivity step from the suffixes
/// declared in the input `.nl`. Returns `None` (quietly) when any
/// required suffix is missing — typical for `.nl` files that aren't
/// sensitivity inputs.
#[allow(clippy::too_many_arguments)]
fn try_compute_sens_step(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    nlp: &Rc<RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    pd: Rc<RefCell<pounce_algorithm::kkt::pd_full_space_solver::PdFullSpaceSolver>>,
    suffixes: &NlSuffixes,
    n_full: usize,
    _m_full: usize,
    x_nominal: &[Number],
) -> Option<Vec<Number>> {
    // Required suffixes. The "_1" suffix tier corresponds to upstream
    // sIPOPT's `n_sens_steps=1` mode. Higher tiers (sens_state_2 etc.)
    // are a Phase-2 follow-up.
    let sens_state = suffixes.var_int.get("sens_state_1")?;
    let sens_state_value = suffixes.var_real.get("sens_state_value_1")?;
    let sens_init_constr = suffixes.con_int.get("sens_init_constr")?;

    if sens_state.len() != n_full || sens_state_value.len() != n_full {
        eprintln!("pounce: sens_state_1 / sens_state_value_1 length mismatch (expected {n_full})");
        return None;
    }

    // Number of parameters and per-parameter (var_idx, constraint_idx)
    // pairs. The integer suffix value is 1..n_params, indexing which
    // parameter slot each variable / constraint maps to.
    let n_params = sens_state.iter().copied().max().unwrap_or(0).max(0) as usize;
    if n_params == 0 {
        return None;
    }

    // For each parameter slot, find its variable index (from
    // sens_state_1) and its pinning-constraint index (from
    // sens_init_constr).
    let mut param_var_idx: Vec<Option<usize>> = vec![None; n_params];
    for (var_idx, &slot) in sens_state.iter().enumerate() {
        if slot > 0 {
            let s = slot as usize;
            if s <= n_params {
                param_var_idx[s - 1] = Some(var_idx);
            }
        }
    }
    let mut param_con_idx: Vec<Option<usize>> = vec![None; n_params];
    for (con_idx, &slot) in sens_init_constr.iter().enumerate() {
        if slot > 0 {
            let s = slot as usize;
            if s <= n_params {
                param_con_idx[s - 1] = Some(con_idx);
            }
        }
    }
    for k in 0..n_params {
        if param_var_idx[k].is_none() || param_con_idx[k].is_none() {
            eprintln!(
                "pounce: parameter {} missing sens_state_1 or sens_init_constr tag",
                k + 1
            );
            return None;
        }
    }

    // Build the SchurData rows: flat compound-vector index for each
    // pinning constraint = n_x + n_s + c_block_idx (i.e. y_c[…] slot).
    // Pounce's compound layout matches upstream's
    // `MetadataMeasurement::GetInitialEqConstraints`
    // (`ref/Ipopt/contrib/sIPOPT/src/SensMetadataMeasurement.cpp:69-83`).
    //
    // Two coordinate transforms are needed when `n_x != n_full` (fixed
    // variables present) or when the c/d split reorders constraints:
    //   * full-x index → var-x index via `IpoptNlp::full_x_to_var_x`
    //   * full-g index → c-block index via `IpoptNlp::full_g_to_c_block`
    let curr = data.borrow().curr.clone()?;
    let n_x = curr.x.dim() as usize;
    let n_s = curr.s.dim() as usize;
    let nlp_ref = nlp.borrow();
    let y_c_offset = n_x + n_s;
    let mut rows: Vec<Index> = Vec::with_capacity(n_params);
    for k in 0..n_params {
        let full_ci = param_con_idx[k].unwrap();
        match nlp_ref.full_g_to_c_block(full_ci as Index) {
            Some(c_idx) => rows.push(y_c_offset as Index + c_idx),
            None => {
                eprintln!(
                    "pounce: parameter {} pinning constraint #{} is an inequality (not in the c block)",
                    k + 1,
                    full_ci
                );
                return None;
            }
        }
    }
    let signs: Vec<Index> = vec![1; n_params];
    let a_data = IndexSchurData::from_parts(rows, signs).ok()?;

    // Δp[k] = perturbed_value - current_value for parameter k. Both
    // sides are read from the user's full-x array (length `n_full`); the
    // caller passes `x_nominal` already lifted via
    // `IpoptNlp::lift_x_to_full`, so indexing by the full-x var index
    // works whether or not other variables were eliminated.
    let mut delta_p: Vec<Number> = Vec::with_capacity(n_params);
    for k in 0..n_params {
        let vi = param_var_idx[k].unwrap();
        delta_p.push(sens_state_value[vi] - x_nominal[vi]);
    }
    drop(nlp_ref);

    let backsolver = PdSensBacksolver::new(data, cq, nlp, pd).ok()?;
    let n_full_pd = backsolver.dim();
    let mut rhs_full = vec![0.0; n_full_pd];
    a_data
        .trans_multiply(&delta_p, &mut rhs_full)
        .map_err(|e| eprintln!("pounce: trans_multiply error: {e:?}"))
        .ok()?;
    let mut dx_full = vec![0.0; n_full_pd];
    if !backsolver.solve(&rhs_full, &mut dx_full) {
        eprintln!("pounce: KKT backsolve failed");
        return None;
    }
    Some(dx_full)
}
