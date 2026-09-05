//! TNLP → IpoptNLP adapter — Phase-3-scope port of
//! `Interfaces/IpTNLPAdapter.{hpp,cpp}`.
//!
//! Splits a user-facing [`TNLP`] (mixed bounds, mixed equality /
//! inequality constraints) into the separated form
//!     min  f(x)
//!     s.t. c(x) = 0    (equality)
//!          d(x) - s = 0,  d_L ≤ s ≤ d_U   (inequality with slacks)
//!          x_L ≤ x ≤ x_U
//! used by the algorithm. This file ships only the **classification**
//! piece — bounds and constraints are sorted into eq/ineq/{lower,upper}
//! sets and the corresponding index maps are computed. The full adapter
//! (function-evaluation routing, sparsity propagation, fixed-variable
//! treatment, scaling) lands with Phase 5 when `IpoptNLP` and
//! `ExpansionMatrix` are wired up.

use crate::tnlp::{BoundsInfo, IndexStyle, Linearity, NlpInfo, TNLP};
use pounce_common::exception::{ExceptionKind, SolverException};
use pounce_common::types::{Index, Number};
use std::cell::RefCell;
use std::rc::Rc;

/// Default infinity threshold for variable / constraint bounds. Matches
/// the `nlp_lower_bound_inf` / `nlp_upper_bound_inf` registered option
/// defaults in upstream Ipopt (`±1e19`).
pub const DEFAULT_NLP_LOWER_BOUND_INF: Number = -1.0e19;
pub const DEFAULT_NLP_UPPER_BOUND_INF: Number = 1.0e19;

/// How a fixed variable (`x_l == x_u`) is handled during classification.
/// Mirrors upstream's `FixedVariableTreatmentEnum` (`IpTNLPAdapter.hpp`).
/// Only the two modes pounce relies on are implemented; `MakeConstraint`
/// and `MakeParameterNoDual` would land alongside their upstream
/// counterparts when needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedVarTreatment {
    /// Default: drop the fixed variable from `x_var` and splice its value
    /// back into `full_x` before user evals (upstream `MAKE_PARAMETER`).
    MakeParameter,
    /// Keep the fixed variable in `x_var` with `x_L == x_U` at the fixed
    /// value; `bound_relax_factor` then widens those tight bounds.
    /// Upstream `RELAX_BOUNDS` (`IpTNLPAdapter.cpp:494-500`).
    RelaxBounds,
}

impl Default for FixedVarTreatment {
    fn default() -> Self {
        Self::MakeParameter
    }
}

/// Sorted decomposition of a TNLP's bounds and constraints. All `*_map`
/// vectors carry **0-based** indices into the full TNLP space.
#[derive(Debug, Clone)]
pub struct BoundClassification {
    pub n_full_x: Index,
    pub n_full_g: Index,
    /// Number of variables with `x_l == x_u` removed from `x_var` under
    /// `fixed_variable_treatment = make_parameter` (the upstream default).
    /// Their indices live in `x_fixed_map` and their values in
    /// `x_fixed_vals`. Zero under `relax_bounds` (fixed vars stay in
    /// `x_var` with tight bounds).
    pub n_x_fixed: Index,
    /// Indices in `[0, n_full_x)` that are not fixed (`x_l < x_u`).
    /// Length is `n_x_var = n_full_x - n_x_fixed`.
    pub x_not_fixed_map: Vec<Index>,
    /// Indices in `[0, n_full_x)` that ARE fixed. Length `n_x_fixed`.
    pub x_fixed_map: Vec<Index>,
    /// Fixed values (== `x_l[i] == x_u[i]`) for each entry of
    /// `x_fixed_map`. Used by `OrigIpoptNlp::lift_x_to_full` to insert
    /// the correct constant into the full-x array before calling the
    /// user's TNLP.
    pub x_fixed_vals: Vec<Number>,
    /// Maps full-x index → var-x index, with `-1` for fixed entries.
    /// Used by sparsity filtering for the Jacobian / Hessian.
    pub full_to_var: Vec<Index>,
    /// Subset of `x_not_fixed_map`'s domain (i.e. positions in `x_var`)
    /// where a finite lower bound is present.
    pub x_l_map: Vec<Index>,
    /// Same for finite upper bounds.
    pub x_u_map: Vec<Index>,
    /// Equality constraint count and indices into `[0, n_full_g)`.
    pub n_c: Index,
    pub c_map: Vec<Index>,
    /// Inequality constraint count and indices into `[0, n_full_g)`.
    pub n_d: Index,
    pub d_map: Vec<Index>,
    /// Subset of `[0, n_d)` with a finite lower bound.
    pub d_l_map: Vec<Index>,
    /// Subset of `[0, n_d)` with a finite upper bound.
    pub d_u_map: Vec<Index>,
    /// Maps full-g index → c-block position, with `-1` for inequality
    /// rows: the O(1) inverse of `c_map`, mirroring `full_to_var`.
    pub full_to_c: Vec<Index>,
    /// Maps full-g index → d-block position, with `-1` for equality
    /// rows: the O(1) inverse of `d_map`, and the exact complement of
    /// `full_to_c` — every row is in one block or the other, so
    /// `(full_to_c[i] < 0) == (full_to_d[i] >= 0)` for every `i`.
    ///
    /// Added so a caller holding a user-space row index can find the
    /// inequality multiplier `y_d` in the compound KKT vector, which
    /// `full_to_c` alone could only refuse to do (pounce#910).
    pub full_to_d: Vec<Index>,
}

impl BoundClassification {
    pub fn n_x_var(&self) -> Index {
        self.x_not_fixed_map.len() as Index
    }
    pub fn n_x_l(&self) -> Index {
        self.x_l_map.len() as Index
    }
    pub fn n_x_u(&self) -> Index {
        self.x_u_map.len() as Index
    }
    pub fn n_d_l(&self) -> Index {
        self.d_l_map.len() as Index
    }
    pub fn n_d_u(&self) -> Index {
        self.d_u_map.len() as Index
    }
}

/// Phase-3 TNLP wrapper. Holds shared ownership of the user's TNLP and
/// the cached problem dimensions / decomposition. Phase 5 will extend
/// this struct with cached scaled/unscaled `f`, `g`, `grad_f`, `jac_g`
/// and a `new_x` flag.
pub struct TNLPAdapter {
    tnlp: Rc<RefCell<dyn TNLP>>,
    info: NlpInfo,
    classification: BoundClassification,
    nlp_lower_bound_inf: Number,
    nlp_upper_bound_inf: Number,
}

impl std::fmt::Debug for TNLPAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TNLPAdapter")
            .field("info", &self.info)
            .field("classification", &self.classification)
            .field("nlp_lower_bound_inf", &self.nlp_lower_bound_inf)
            .field("nlp_upper_bound_inf", &self.nlp_upper_bound_inf)
            .finish_non_exhaustive()
    }
}

impl TNLPAdapter {
    /// Construct an adapter from a TNLP. Reads `get_nlp_info` and
    /// `get_bounds_info`, performs bound + constraint classification,
    /// and stores the result. Uses the default `±1e19` infinity
    /// thresholds.
    pub fn new(tnlp: Rc<RefCell<dyn TNLP>>) -> Result<Self, SolverException> {
        Self::new_with_options(
            tnlp,
            DEFAULT_NLP_LOWER_BOUND_INF,
            DEFAULT_NLP_UPPER_BOUND_INF,
            FixedVarTreatment::MakeParameter,
        )
    }

    /// Construct an adapter with custom infinity thresholds (the user
    /// can override these via `nlp_lower_bound_inf` / `nlp_upper_bound_inf`).
    pub fn new_with_inf(
        tnlp: Rc<RefCell<dyn TNLP>>,
        nlp_lower_bound_inf: Number,
        nlp_upper_bound_inf: Number,
    ) -> Result<Self, SolverException> {
        Self::new_with_options(
            tnlp,
            nlp_lower_bound_inf,
            nlp_upper_bound_inf,
            FixedVarTreatment::MakeParameter,
        )
    }

    /// Construct an adapter with custom infinity thresholds and an
    /// explicit `fixed_variable_treatment`. Mirrors upstream
    /// `IpTNLPAdapter::ProcessOptions` + `Initialize` (`IpTNLPAdapter.cpp:240`,
    /// `:430-633`): when `MakeParameter` would leave fewer free variables
    /// than equality constraints, automatically retry classification with
    /// `RelaxBounds` (`IpTNLPAdapter.cpp:623-633`).
    pub fn new_with_options(
        tnlp: Rc<RefCell<dyn TNLP>>,
        nlp_lower_bound_inf: Number,
        nlp_upper_bound_inf: Number,
        fixed_var_treatment: FixedVarTreatment,
    ) -> Result<Self, SolverException> {
        if nlp_lower_bound_inf >= nlp_upper_bound_inf {
            return Err(SolverException::new(
                ExceptionKind::OPTION_INVALID,
                "Option \"nlp_lower_bound_inf\" must be smaller than \
                 \"nlp_upper_bound_inf\".",
                file!(),
                line!() as Index,
            ));
        }

        let info = {
            let mut t = tnlp.borrow_mut();
            t.get_nlp_info().ok_or_else(|| {
                SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "TNLP::get_nlp_info returned None.",
                    file!(),
                    line!() as Index,
                )
            })?
        };

        if info.n <= 0 {
            return Err(SolverException::new(
                ExceptionKind::INVALID_TNLP,
                format!("TNLP::get_nlp_info reported n = {} (must be > 0).", info.n),
                file!(),
                line!() as Index,
            ));
        }
        if info.m < 0 {
            return Err(SolverException::new(
                ExceptionKind::INVALID_TNLP,
                format!("TNLP::get_nlp_info reported m = {} (must be ≥ 0).", info.m),
                file!(),
                line!() as Index,
            ));
        }

        let n_full_x = info.n;
        let n_full_g = info.m;

        let mut x_l = vec![0.0; n_full_x as usize];
        let mut x_u = vec![0.0; n_full_x as usize];
        let mut g_l = vec![0.0; n_full_g as usize];
        let mut g_u = vec![0.0; n_full_g as usize];

        {
            let mut t = tnlp.borrow_mut();
            let ok = t.get_bounds_info(BoundsInfo {
                x_l: &mut x_l,
                x_u: &mut x_u,
                g_l: &mut g_l,
                g_u: &mut g_u,
            });
            if !ok {
                return Err(SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "TNLP::get_bounds_info returned false.",
                    file!(),
                    line!() as Index,
                ));
            }
        }

        let mut treatment = fixed_var_treatment;
        let mut classification = classify_bounds(
            n_full_x,
            n_full_g,
            &x_l,
            &x_u,
            &g_l,
            &g_u,
            nlp_lower_bound_inf,
            nlp_upper_bound_inf,
            treatment,
        )?;

        // Mirror upstream `IpTNLPAdapter.cpp:623-633`: if `make_parameter`
        // dropped enough variables to leave `n_x_var < n_c`, automatically
        // switch to `relax_bounds` (keep fixed vars in the active set with
        // tight bounds) and redo classification. Without this, square /
        // over-determined-after-fixing problems abort with
        // `NotEnoughDegreesOfFreedom`.
        if treatment == FixedVarTreatment::MakeParameter
            && classification.n_x_fixed > 0
            && classification.n_x_var() > 0
            && classification.n_x_var() < classification.n_c
        {
            treatment = FixedVarTreatment::RelaxBounds;
            classification = classify_bounds(
                n_full_x,
                n_full_g,
                &x_l,
                &x_u,
                &g_l,
                &g_u,
                nlp_lower_bound_inf,
                nlp_upper_bound_inf,
                treatment,
            )?;
        }

        Ok(Self {
            tnlp,
            info,
            classification,
            nlp_lower_bound_inf,
            nlp_upper_bound_inf,
        })
    }

    pub fn nlp_info(&self) -> &NlpInfo {
        &self.info
    }

    pub fn classification(&self) -> &BoundClassification {
        &self.classification
    }

    pub fn nlp_lower_bound_inf(&self) -> Number {
        self.nlp_lower_bound_inf
    }

    pub fn nlp_upper_bound_inf(&self) -> Number {
        self.nlp_upper_bound_inf
    }

    pub fn tnlp(&self) -> &Rc<RefCell<dyn TNLP>> {
        &self.tnlp
    }

    /// Which variables the limited-memory quasi-Newton approximation
    /// should span — port of `IpTNLPAdapter::GetQuasiNewtonApproxSpaces`
    /// (`IpTNLPAdapter.cpp:2330`), the Ipopt hook behind CasADi's
    /// `pass_nonlinear_variables` (gh#624).
    ///
    /// Returns positions in the algorithm's **compressed `x_var` space**
    /// (fixed variables already removed), sorted and deduplicated.
    /// `None` means "every variable is nonlinear" — the default, and the
    /// identity case upstream signals with a NULL expansion matrix.
    ///
    /// Precedence mirrors upstream: a TNLP that implements
    /// `get_number_of_nonlinear_variables` wins; `num_linear_variables`
    /// is the contiguous-prefix fallback consulted only when the
    /// callback declines (returns a negative count).
    pub fn quasi_newton_nonlinear_vars(
        &self,
        num_linear_variables: Index,
    ) -> Result<Option<Vec<Index>>, SolverException> {
        let n_full_x = self.classification.n_full_x;
        let num_nonlin = self.tnlp.borrow_mut().get_number_of_nonlinear_variables();

        let full_positions: Vec<Index> = if num_nonlin < 0 {
            // No callback information. Upstream then treats the first
            // `num_linear_variables` variables as linear and everything
            // after them as nonlinear; `0` (the default) means "all
            // nonlinear", i.e. no restriction at all.
            if num_linear_variables <= 0 {
                return Ok(None);
            }
            if num_linear_variables > n_full_x {
                return Err(SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "num_linear_variables exceeds the number of variables",
                    file!(),
                    line!() as Index,
                ));
            }
            (num_linear_variables..n_full_x).collect()
        } else {
            if num_nonlin > n_full_x {
                return Err(SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "TNLP's get_number_of_nonlinear_variables exceeds the number of variables",
                    file!(),
                    line!() as Index,
                ));
            }
            let mut pos = vec![0 as Index; num_nonlin as usize];
            if !self
                .tnlp
                .borrow_mut()
                .get_list_of_nonlinear_variables(&mut pos)
            {
                return Err(SolverException::new(
                    ExceptionKind::INVALID_TNLP,
                    "TNLP's get_number_of_nonlinear_variables returns a non-negative number, \
                     but get_list_of_nonlinear_variables returns false",
                    file!(),
                    line!() as Index,
                ));
            }
            // The list arrives in the TNLP's own index style.
            let offset = match self.info.index_style {
                IndexStyle::C => 0,
                IndexStyle::Fortran => 1,
            };
            for p in pos.iter_mut() {
                *p -= offset;
                if *p < 0 || *p >= n_full_x {
                    return Err(SolverException::new(
                        ExceptionKind::INVALID_TNLP,
                        "TNLP's get_list_of_nonlinear_variables returned an out-of-range index",
                        file!(),
                        line!() as Index,
                    ));
                }
            }
            pos
        };

        // Drop fixed variables and translate to the compressed space —
        // upstream filters through `P_x_full_x_` the same way, so a mask
        // stated over the user's variables survives
        // `fixed_variable_treatment = make_parameter`.
        let mut small: Vec<Index> = full_positions
            .iter()
            .filter_map(|&i| {
                let v = self.classification.full_to_var[i as usize];
                (v >= 0).then_some(v)
            })
            .collect();
        small.sort_unstable();
        small.dedup();

        // Everything nonlinear ⇒ no restriction (and no expansion matrix).
        if small.len() as Index == self.classification.n_x_var() {
            return Ok(None);
        }
        Ok(Some(small))
    }

    /// Which variables the **objective's** second derivatives can reach,
    /// in the compressed `x_var` space, sorted and deduplicated.
    ///
    /// This is the objective element's support for the partitioned
    /// quasi-Newton Hessian
    /// ([`pounce_algorithm::hess::partitioned_quasi_newton`]). Every
    /// constraint element takes its support from a row of the Jacobian,
    /// whose pattern the TNLP must declare — but nothing in the TNLP
    /// contract declares the objective's. Reading it off the nonzeros of
    /// the first `∇f` instead makes the pattern *value-derived*: a
    /// coordinate whose `∂f/∂x_i` happens to vanish at the starting
    /// point is excluded for the whole solve. Measured on
    /// `benchmarks/large_scale` `laptime`, which declares 321 objective
    /// gradient nonzeros, that heuristic captured 161.
    ///
    /// `get_objective_variables_linearity` is the structural answer and
    /// is a pounce extension that already exists for exactly this class
    /// of question (`pounce-nl` implements it). `None` means the TNLP
    /// declines, and the caller falls back to the gradient nonzeros.
    pub fn objective_nonlinear_vars(&self) -> Option<Vec<Index>> {
        let n_full_x = self.classification.n_full_x;
        let mut types = vec![Linearity::Linear; n_full_x as usize];
        if !self
            .tnlp
            .borrow_mut()
            .get_objective_variables_linearity(&mut types)
        {
            return None;
        }
        let mut small: Vec<Index> = types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == Linearity::NonLinear)
            .filter_map(|(i, _)| {
                let v = self.classification.full_to_var[i];
                (v >= 0).then_some(v)
            })
            .collect();
        small.sort_unstable();
        small.dedup();
        Some(small)
    }
}

/// Split the full variable / constraint sets into fixed vs. free variables and
/// equality vs. inequality rows, recording which sides carry a real bound.
///
/// **Deliberate divergence from upstream (gh #398).** `IpTNLPAdapter` tests
/// `lower == upper` and `lower > upper` on the *raw* bound pair, before asking
/// whether either side is present, and only consults `nlp_lower_bound_inf` /
/// `nlp_upper_bound_inf` afterwards. That is safe only while every real bound
/// sits inside the sentinels. A `<=`-only row arrives with its absent lower
/// bound filled in at `-1e19`; if the row's genuine upper bound is more
/// negative than that (`-5e20` is perfectly ordinary, and both sentinels are
/// user-settable options besides), the raw pair reads as crossed and a feasible
/// model is rejected as `Invalid_Problem_Definition`.
///
/// So presence is decided first, and *directionally* — a lower bound is absent
/// at or below `lo_inf`, an upper bound at or above `up_inf`, the convention
/// `pounce_presolve::bound_tighten` already uses. Equality, fixed-variable, and
/// crossed-pair tests then run on the present bounds only, which leaves
/// `INCONSISTENT_BOUNDS` for what it is meant for: a modeller who declared both
/// sides and crossed them. Models upstream accepts classify identically; the
/// divergence is confined to bounds outside the sentinels, which upstream
/// cannot express at all.
#[allow(clippy::too_many_arguments)]
fn classify_bounds(
    n_full_x: Index,
    n_full_g: Index,
    x_l: &[Number],
    x_u: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    lo_inf: Number,
    up_inf: Number,
    treatment: FixedVarTreatment,
) -> Result<BoundClassification, SolverException> {
    let nx = n_full_x as usize;
    let ng = n_full_g as usize;

    // --- Variables ---------------------------------------------------
    let mut x_not_fixed_map: Vec<Index> = Vec::with_capacity(nx);
    let mut x_fixed_map: Vec<Index> = Vec::new();
    let mut x_fixed_vals: Vec<Number> = Vec::new();
    let mut full_to_var: Vec<Index> = vec![-1; nx];
    let mut x_l_map: Vec<Index> = Vec::new();
    let mut x_u_map: Vec<Index> = Vec::new();
    let mut n_x_fixed: Index = 0;

    for i in 0..nx {
        let lo = x_l[i];
        let hi = x_u[i];
        // Presence is *directional*, not a symmetric magnitude test: a lower
        // bound is absent only at or below `lo_inf`, an upper bound only at or
        // above `up_inf`. A finite bound past the opposite sentinel (say an
        // upper bound of -5e20) is an ordinary bound, not an "infinite" one, so
        // it must not be compared against the absent side's sentinel value.
        let lo_present = lo > lo_inf;
        let hi_present = hi < up_inf;
        if lo_present && hi_present && lo > hi {
            return Err(SolverException::new(
                ExceptionKind::INCONSISTENT_BOUNDS,
                format!(
                    "There are inconsistent bounds on variable {i}: lower = {lo:25.16e} \
                     and upper = {hi:25.16e}."
                ),
                file!(),
                line!() as Index,
            ));
        }
        if lo_present && hi_present && lo == hi {
            match treatment {
                FixedVarTreatment::MakeParameter => {
                    // Drop fixed vars from x_var entirely. Their values are
                    // spliced back into the full-x array each time we call
                    // into the user's TNLP (see `OrigIpoptNlp::lift_x_to_full`).
                    n_x_fixed += 1;
                    x_fixed_map.push(i as Index);
                    x_fixed_vals.push(lo);
                    continue;
                }
                FixedVarTreatment::RelaxBounds => {
                    // Keep the var in the active set with tight bounds on
                    // both sides; `OrigIpoptNlp::relax_bounds` will widen
                    // them by `bound_relax_factor`. Matches upstream
                    // `IpTNLPAdapter.cpp:494-500`.
                    let var_idx = x_not_fixed_map.len() as Index;
                    x_not_fixed_map.push(i as Index);
                    full_to_var[i] = var_idx;
                    x_l_map.push(var_idx);
                    x_u_map.push(var_idx);
                    continue;
                }
            }
        }
        let var_idx = x_not_fixed_map.len() as Index;
        x_not_fixed_map.push(i as Index);
        full_to_var[i] = var_idx;
        if lo_present {
            x_l_map.push(var_idx);
        }
        if hi_present {
            x_u_map.push(var_idx);
        }
    }

    // --- Constraints -------------------------------------------------
    let mut c_map: Vec<Index> = Vec::new();
    let mut d_map: Vec<Index> = Vec::new();
    let mut d_l_map: Vec<Index> = Vec::new();
    let mut d_u_map: Vec<Index> = Vec::new();
    let mut full_to_c: Vec<Index> = vec![-1; ng];
    let mut full_to_d: Vec<Index> = vec![-1; ng];

    for i in 0..ng {
        let lo = g_l[i];
        let hi = g_u[i];
        // Same directional presence test as the variable box above. A
        // `<=`-only row arrives with `g_l` at the `-1e19` sentinel; if its real
        // upper bound is more negative than that (a legitimate `-5e20`), the
        // pair only looks crossed under a symmetric reading of the sentinel.
        let lo_present = lo > lo_inf;
        let hi_present = hi < up_inf;
        if lo_present && hi_present {
            if lo > hi {
                return Err(SolverException::new(
                    ExceptionKind::INCONSISTENT_BOUNDS,
                    format!(
                        "There are inconsistent bounds on constraint function {i}: \
                         lower = {lo:25.16e} and upper = {hi:25.16e}."
                    ),
                    file!(),
                    line!() as Index,
                ));
            }
            if lo == hi {
                full_to_c[i] = c_map.len() as Index;
                c_map.push(i as Index);
                continue;
            }
        }
        let d_idx = d_map.len() as Index;
        full_to_d[i] = d_idx;
        d_map.push(i as Index);
        if lo_present {
            d_l_map.push(d_idx);
        }
        if hi_present {
            d_u_map.push(d_idx);
        }
    }

    let n_c = c_map.len() as Index;
    let n_d = d_map.len() as Index;

    Ok(BoundClassification {
        n_full_x,
        n_full_g,
        n_x_fixed,
        x_not_fixed_map,
        x_fixed_map,
        x_fixed_vals,
        full_to_var,
        x_l_map,
        x_u_map,
        n_c,
        c_map,
        n_d,
        d_map,
        d_l_map,
        d_u_map,
        full_to_c,
        full_to_d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tnlp::{IndexStyle, IpoptCq, IpoptData, Solution, SparsityRequest, StartingPoint};

    /// HS071: min x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    /// s.t.   x[0]*x[1]*x[2]*x[3] >= 25                (inequality)
    ///        x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2 == 40  (equality)
    ///        1 <= x[i] <= 5
    struct Hs071;
    impl TNLP for Hs071 {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 4,
                m: 2,
                nnz_jac_g: 8,
                nnz_h_lag: 10,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[1.0; 4]);
            b.x_u.copy_from_slice(&[5.0; 4]);
            // Constraint 0: 25 <= g_0 <= +inf  (inequality, finite lower only)
            // Constraint 1: 40 == g_1 == 40    (equality)
            b.g_l.copy_from_slice(&[25.0, 40.0]);
            b.g_u.copy_from_slice(&[2.0e19, 40.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            if let SparsityRequest::Structure { irow, jcol } = mode {
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn hs071_decomposes_to_one_eq_one_ineq() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071));
        let adapter = TNLPAdapter::new(tnlp).unwrap();
        let c = adapter.classification();
        assert_eq!(c.n_full_x, 4);
        assert_eq!(c.n_full_g, 2);
        assert_eq!(c.n_x_fixed, 0);
        assert_eq!(c.n_x_var(), 4);
        assert!(c.x_fixed_map.is_empty());
        assert_eq!(c.full_to_var, vec![0, 1, 2, 3]);
        // All four variables have both finite bounds.
        assert_eq!(c.x_l_map, vec![0, 1, 2, 3]);
        assert_eq!(c.x_u_map, vec![0, 1, 2, 3]);
        // Constraint #0 is the inequality, #1 is the equality.
        assert_eq!(c.n_c, 1);
        assert_eq!(c.c_map, vec![1]);
        assert_eq!(c.n_d, 1);
        assert_eq!(c.d_map, vec![0]);
        // The single inequality has a finite lower bound (25) and an
        // infinite upper bound (2e19 == nlp_upper_bound_inf).
        assert_eq!(c.d_l_map, vec![0]);
        assert!(c.d_u_map.is_empty());
        assert_eq!(adapter.nlp_info().nnz_jac_g, 8);
    }

    /// Variant with one fixed variable (x[0] in [3,3]) and one free
    /// variable (x[1] in [-inf, +inf]) to exercise the bound-only and
    /// fixed paths.
    struct Mixed;
    impl TNLP for Mixed {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 2,
                nnz_jac_g: 6,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            // x[0] fixed at 3, x[1] free, x[2] upper-only at 7.
            b.x_l.copy_from_slice(&[3.0, -2.0e19, -2.0e19]);
            b.x_u.copy_from_slice(&[3.0, 2.0e19, 7.0]);
            // g[0]: 0 <= ... <= 1 (two-sided ineq)
            // g[1]: free constraint (-inf, +inf) — still classified as ineq.
            b.g_l.copy_from_slice(&[0.0, -2.0e19]);
            b.g_u.copy_from_slice(&[1.0, 2.0e19]);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn mixed_bounds_classifies_correctly() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mixed));
        let adapter = TNLPAdapter::new(tnlp).unwrap();
        let c = adapter.classification();
        assert_eq!(c.n_full_x, 3);
        assert_eq!(c.n_x_fixed, 1);
        // x[0] fixed at 3 → removed from x_var (make_parameter).
        // x[1] free, x[2] upper-only → both in x_var.
        assert_eq!(c.n_x_var(), 2);
        assert_eq!(c.x_not_fixed_map, vec![1, 2]);
        assert_eq!(c.x_fixed_map, vec![0]);
        assert_eq!(c.x_fixed_vals, vec![3.0]);
        assert_eq!(c.full_to_var, vec![-1, 0, 1]);
        // After fixed-var removal, x[1] (now var idx 0) is fully free,
        // x[2] (now var idx 1) has only an upper bound.
        assert!(c.x_l_map.is_empty());
        assert_eq!(c.x_u_map, vec![1]);
        // No equalities; both constraints are classified as inequalities.
        assert_eq!(c.n_c, 0);
        assert_eq!(c.n_d, 2);
        assert_eq!(c.d_map, vec![0, 1]);
        // d[0] has finite lower (0) and finite upper (1).
        // d[1] is fully free — neither bound finite.
        assert_eq!(c.d_l_map, vec![0]);
        assert_eq!(c.d_u_map, vec![0]);
    }

    /// Inconsistent bounds (lo > hi) should error.
    struct Bad;
    impl TNLP for Bad {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = 5.0;
            b.x_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    /// Two free vars, one fixed var, and two equality constraints
    /// (`n_full_x=3`, `n_full_g=2`). Under `make_parameter` the fixed var
    /// is dropped, leaving `n_x_var=2 == n_c=2` (boundary OK — the gate
    /// trips on `<`, not `<=`). Under `relax_bounds` the fixed var stays
    /// in `x_var` with tight bounds.
    struct OneFixedTwoEq;
    impl TNLP for OneFixedTwoEq {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 2,
                nnz_jac_g: 6,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[2.5, -2.0e19, -2.0e19]);
            b.x_u.copy_from_slice(&[2.5, 2.0e19, 2.0e19]);
            b.g_l.copy_from_slice(&[0.0, 0.0]);
            b.g_u.copy_from_slice(&[0.0, 0.0]);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn relax_bounds_keeps_fixed_var_in_active_set() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneFixedTwoEq));
        let adapter = TNLPAdapter::new_with_options(
            tnlp,
            DEFAULT_NLP_LOWER_BOUND_INF,
            DEFAULT_NLP_UPPER_BOUND_INF,
            FixedVarTreatment::RelaxBounds,
        )
        .unwrap();
        let c = adapter.classification();
        assert_eq!(c.n_full_x, 3);
        assert_eq!(c.n_x_fixed, 0, "relax_bounds keeps fixed var in x_var");
        assert_eq!(c.n_x_var(), 3);
        assert_eq!(c.x_not_fixed_map, vec![0, 1, 2]);
        assert!(c.x_fixed_map.is_empty());
        assert!(c.x_fixed_vals.is_empty());
        assert_eq!(c.full_to_var, vec![0, 1, 2]);
        // The fixed var (index 0) gets tight finite bounds; the other two
        // are infinite both sides.
        assert_eq!(c.x_l_map, vec![0]);
        assert_eq!(c.x_u_map, vec![0]);
        assert_eq!(c.n_c, 2);
    }

    /// Same problem, default `make_parameter` treatment: `n_x_var = 2`,
    /// `n_c = 2` — no auto-retry triggers (boundary `n_x_var == n_c`).
    #[test]
    fn make_parameter_no_retry_when_boundary_dof() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneFixedTwoEq));
        let adapter = TNLPAdapter::new(tnlp).unwrap();
        let c = adapter.classification();
        assert_eq!(c.n_x_fixed, 1);
        assert_eq!(c.n_x_var(), 2);
        assert_eq!(c.n_c, 2);
    }

    /// Powerflow-style: one free var, two fixed vars, two equality
    /// constraints. Under default `make_parameter`, `n_x_var = 1 < n_c = 2`
    /// would trip the DOF gate — adapter must auto-retry with
    /// `relax_bounds` so all three vars stay active.
    struct DofRescue;
    impl TNLP for DofRescue {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 2,
                nnz_jac_g: 6,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[2.5, 1.0, -2.0e19]);
            b.x_u.copy_from_slice(&[2.5, 1.0, 2.0e19]);
            b.g_l.copy_from_slice(&[0.0, 0.0]);
            b.g_u.copy_from_slice(&[0.0, 0.0]);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn make_parameter_auto_retries_with_relax_bounds() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(DofRescue));
        let adapter = TNLPAdapter::new(tnlp).unwrap();
        let c = adapter.classification();
        // Auto-retry kicked in: classification matches relax_bounds, not
        // the (failing) make_parameter result.
        assert_eq!(c.n_x_fixed, 0);
        assert_eq!(c.n_x_var(), 3);
        assert_eq!(c.x_not_fixed_map, vec![0, 1, 2]);
        // Both fixed vars get tight finite bounds.
        assert_eq!(c.x_l_map, vec![0, 1]);
        assert_eq!(c.x_u_map, vec![0, 1]);
        assert_eq!(c.n_c, 2);
    }

    #[test]
    fn inconsistent_variable_bounds_is_rejected() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Bad));
        let err = TNLPAdapter::new(tnlp).unwrap_err();
        assert_eq!(err.kind, ExceptionKind::INCONSISTENT_BOUNDS);
    }

    /// A TNLP that reports whatever bounds it is handed and nothing else —
    /// enough to drive `classify_bounds` through the adapter constructor.
    struct BoundsOnly {
        x_l: Vec<Number>,
        x_u: Vec<Number>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
    }
    impl TNLP for BoundsOnly {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: self.x_l.len() as Index,
                m: self.g_l.len() as Index,
                nnz_jac_g: 0,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&self.x_l);
            b.x_u.copy_from_slice(&self.x_u);
            b.g_l.copy_from_slice(&self.g_l);
            b.g_u.copy_from_slice(&self.g_u);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn classify(b: BoundsOnly) -> Result<BoundClassification, SolverException> {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(b));
        TNLPAdapter::new(tnlp).map(|a| a.classification().clone())
    }

    /// **gh #398.** A `<=`-only row whose real upper bound is *more negative*
    /// than the absent-lower sentinel is an ordinary one-sided row, not a
    /// crossed pair.
    ///
    /// `.nl` fills the absent lower bound of `-1e30·x <= -5e20` with the
    /// `-1e19` sentinel, so a symmetric magnitude reading sees
    /// `-1e19 > -5e20` and calls the pair inconsistent — `504
    /// Invalid_Problem_Definition` on a model whose declared `x0` sits at
    /// exactly zero violation. Presence is directional: a *lower* bound is
    /// absent only at or below `nlp_lower_bound_inf`, so the sentinel is not a
    /// bound to compare against at all.
    #[test]
    fn one_sided_row_beyond_the_sentinel_is_not_a_crossed_pair() {
        let c = classify(BoundsOnly {
            x_l: vec![-2.0e19],
            x_u: vec![2.0e19],
            g_l: vec![DEFAULT_NLP_LOWER_BOUND_INF],
            g_u: vec![-5.0000000000000007e20],
        })
        .expect("a one-sided row at -5e20 must classify, not error");
        assert_eq!(c.n_c, 0, "not an equality row");
        assert_eq!(c.n_d, 1);
        assert!(
            c.d_l_map.is_empty(),
            "the -1e19 lower bound is the absent-bound sentinel"
        );
        assert_eq!(c.d_u_map, vec![0], "-5e20 is a real, present upper bound");
    }

    /// The mirror image on the variable box: a lower bound past `+INF` with no
    /// upper bound. Symmetrically read, `+5e20 > +1e19` looked like a crossed
    /// box (and `lo == hi` at the sentinel looked like a *fixed* variable);
    /// directionally it is a lower-bounded-only variable.
    #[test]
    fn one_sided_var_bound_beyond_the_sentinel_is_not_crossed() {
        let c = classify(BoundsOnly {
            x_l: vec![5.0e20],
            x_u: vec![DEFAULT_NLP_UPPER_BOUND_INF],
            g_l: vec![],
            g_u: vec![],
        })
        .expect("x >= 5e20 with no upper bound must classify, not error");
        assert_eq!(c.n_x_fixed, 0, "an absent upper bound does not fix the var");
        assert_eq!(c.n_x_var(), 1);
        assert_eq!(c.x_l_map, vec![0]);
        assert!(c.x_u_map.is_empty());
    }

    /// The guard that must survive the fix: when *both* bounds are present and
    /// crossed, that is a genuine modelling error and still an error here.
    #[test]
    fn genuinely_crossed_present_bounds_are_still_rejected() {
        let err = classify(BoundsOnly {
            x_l: vec![-2.0e19],
            x_u: vec![2.0e19],
            g_l: vec![5.0],
            g_u: vec![3.0],
        })
        .expect_err("5 <= g <= 3 is inconsistent");
        assert_eq!(err.kind, ExceptionKind::INCONSISTENT_BOUNDS);

        let err = classify(BoundsOnly {
            x_l: vec![5.0],
            x_u: vec![3.0],
            g_l: vec![],
            g_u: vec![],
        })
        .expect_err("x in [5, 3] is inconsistent");
        assert_eq!(err.kind, ExceptionKind::INCONSISTENT_BOUNDS);
    }

    /// Equality detection is likewise gated on both bounds being present: two
    /// equal bounds that are *both* the same sentinel value describe a
    /// one-sided row, not an equality. `g_l = g_u = 1e20` is `g >= 1e20`.
    #[test]
    fn equal_bounds_past_the_sentinel_are_one_sided_not_an_equality() {
        let c = classify(BoundsOnly {
            x_l: vec![-2.0e19],
            x_u: vec![2.0e19],
            g_l: vec![1.0e20],
            g_u: vec![1.0e20],
        })
        .expect("classify");
        assert_eq!(
            c.n_c, 0,
            "the upper bound is absent, so this is no equality"
        );
        assert_eq!(c.n_d, 1);
        assert_eq!(c.d_l_map, vec![0]);
        assert!(c.d_u_map.is_empty());
    }

    // ---- gh#624: nonlinear-variable subsets for the L-BFGS Hessian ----

    /// Same 3-variable model as [`Mixed`] (x[0] fixed at 3), with the
    /// two Ipopt nonlinear-variable callbacks under test control.
    struct NonlinVars {
        style: IndexStyle,
        num: Index,
        list: Vec<Index>,
        list_ok: bool,
    }

    impl NonlinVars {
        fn declaring(list: &[Index]) -> Self {
            Self {
                style: IndexStyle::C,
                num: list.len() as Index,
                list: list.to_vec(),
                list_ok: true,
            }
        }
        fn silent() -> Self {
            Self {
                style: IndexStyle::C,
                num: -1,
                list: Vec::new(),
                list_ok: false,
            }
        }
    }

    impl TNLP for NonlinVars {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 2,
                nnz_jac_g: 6,
                nnz_h_lag: 0,
                index_style: self.style,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[3.0, -2.0e19, -2.0e19]);
            b.x_u.copy_from_slice(&[3.0, 2.0e19, 7.0]);
            b.g_l.copy_from_slice(&[0.0, -2.0e19]);
            b.g_u.copy_from_slice(&[1.0, 2.0e19]);
            true
        }
        fn get_starting_point(&mut self, _sp: StartingPoint<'_>) -> bool {
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.fill(0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            _m: SparsityRequest<'_>,
        ) -> bool {
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}

        fn get_number_of_nonlinear_variables(&mut self) -> Index {
            self.num
        }
        fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
            if !self.list_ok {
                return false;
            }
            pos.copy_from_slice(&self.list);
            true
        }
    }

    fn adapter_for(t: NonlinVars) -> TNLPAdapter {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(t));
        TNLPAdapter::new(tnlp).unwrap()
    }

    #[test]
    fn silent_tnlp_and_no_option_means_all_nonlinear() {
        let adapter = adapter_for(NonlinVars::silent());
        assert_eq!(adapter.quasi_newton_nonlinear_vars(0).unwrap(), None);
    }

    #[test]
    fn declared_subset_maps_through_fixed_variable_removal() {
        // x[0] is fixed, so it leaves the algorithm's space entirely;
        // x[2] is var-index 1. Declaring {x[0], x[2]} must come out as
        // the single compressed position 1 — not 2, and not a
        // dangling reference to the removed variable.
        let adapter = adapter_for(NonlinVars::declaring(&[0, 2]));
        assert_eq!(
            adapter.quasi_newton_nonlinear_vars(0).unwrap(),
            Some(vec![1])
        );
    }

    #[test]
    fn declared_subset_is_read_in_the_tnlp_index_style() {
        let mut t = NonlinVars::declaring(&[3]); // 1-based ⇒ x[2]
        t.style = IndexStyle::Fortran;
        let adapter = adapter_for(t);
        assert_eq!(
            adapter.quasi_newton_nonlinear_vars(0).unwrap(),
            Some(vec![1])
        );
    }

    #[test]
    fn declaring_every_free_variable_is_the_identity_case() {
        // {x[1], x[2]} is the whole compressed space ⇒ no restriction,
        // and no expansion matrix downstream.
        let adapter = adapter_for(NonlinVars::declaring(&[1, 2]));
        assert_eq!(adapter.quasi_newton_nonlinear_vars(0).unwrap(), None);
    }

    #[test]
    fn unsorted_and_duplicated_declarations_are_normalized() {
        let adapter = adapter_for(NonlinVars::declaring(&[2, 2, 0]));
        assert_eq!(
            adapter.quasi_newton_nonlinear_vars(0).unwrap(),
            Some(vec![1])
        );
    }

    #[test]
    fn num_linear_variables_is_the_fallback_prefix() {
        // No callback information: the first two variables are linear,
        // leaving full-x {2} ⇒ compressed {1}.
        let adapter = adapter_for(NonlinVars::silent());
        assert_eq!(
            adapter.quasi_newton_nonlinear_vars(2).unwrap(),
            Some(vec![1])
        );
    }

    #[test]
    fn callback_information_beats_num_linear_variables() {
        // Upstream precedence: with the callback answered,
        // `num_linear_variables` is ignored outright.
        let adapter = adapter_for(NonlinVars::declaring(&[0, 2]));
        assert_eq!(
            adapter.quasi_newton_nonlinear_vars(2).unwrap(),
            Some(vec![1])
        );
    }

    #[test]
    fn refusing_the_list_after_declaring_a_count_is_an_error() {
        let mut t = NonlinVars::declaring(&[2]);
        t.list_ok = false;
        let adapter = adapter_for(t);
        assert!(adapter.quasi_newton_nonlinear_vars(0).is_err());
    }

    #[test]
    fn out_of_range_declarations_are_rejected() {
        let adapter = adapter_for(NonlinVars::declaring(&[7]));
        assert!(adapter.quasi_newton_nonlinear_vars(0).is_err());

        let adapter = adapter_for(NonlinVars::silent());
        assert!(adapter.quasi_newton_nonlinear_vars(99).is_err());
    }

    // ── the c/d split is a partition (jkitchin/pounce#910) ──────────
    //
    // `full_to_d` is documented as the exact complement of
    // `full_to_c`, and `python/pounce/sensitivity/_session.py`'s
    // `mult_entry` leans on it: it asks the equality block, then the
    // inequality block, and treats both answering `None` as
    // unreachable. That claim is about THIS function, so it is pinned
    // here rather than asserted downstream.
    //
    // The row worth naming is the free one. A `-inf <= g <= inf` row
    // is the shape a reader expects to fall through both blocks, and
    // Pyomo cannot even construct one, so no end-to-end test can
    // reach it -- which is exactly why the premise belongs at this
    // level. It lands in `d` with empty bound-map entries, and the
    // gate downstream then refuses it as inactive, which is the
    // truth: a free row's multiplier is zero always.

    fn split_of(g_l: &[Number], g_u: &[Number]) -> BoundClassification {
        classify_bounds(
            1,
            g_l.len() as Index,
            &[-1e19],
            &[1e19],
            g_l,
            g_u,
            -1e19,
            1e19,
            FixedVarTreatment::MakeParameter,
        )
        .unwrap()
    }

    #[test]
    fn every_constraint_row_lands_in_exactly_one_of_c_and_d() {
        // equality, <=-only, >=-only, two-sided range, and free.
        let cls = split_of(
            &[2.0, -1e19, 0.0, -1.0, -1e19],
            &[2.0, 5.0, 1e19, 1.0, 1e19],
        );
        for i in 0..5usize {
            let in_c = cls.full_to_c[i] >= 0;
            let in_d = cls.full_to_d[i] >= 0;
            assert!(
                in_c != in_d,
                "row {i} is in c={in_c} and d={in_d}; the split must be a \
                 partition, so `mult_entry`'s both-None branch stays \
                 unreachable",
            );
        }
        assert_eq!(cls.full_to_c, vec![0, -1, -1, -1, -1]);
        assert_eq!(cls.full_to_d, vec![-1, 0, 1, 2, 3]);
    }

    #[test]
    fn a_free_row_gets_a_d_slot_with_no_bound_on_either_side() {
        let cls = split_of(&[-1e19], &[1e19]);
        assert_eq!(cls.full_to_c, vec![-1], "a free row is not an equality");
        assert_eq!(cls.full_to_d[0], 0, "but it does get a d slot");
        assert_eq!(cls.d_map, vec![0]);
        assert!(
            cls.d_l_map.is_empty() && cls.d_u_map.is_empty(),
            "and no bound on either side, so nothing can be active",
        );
    }

    #[test]
    fn full_to_d_is_the_positional_inverse_of_d_map() {
        // The accessor indexes the `y_d` block directly, so an
        // off-by-one here is a neighbouring row's multiplier -- the
        // silent failure the newtype work (gh#764 item 3) is about.
        let cls = split_of(&[0.0, 1.0, 1.0, -1e19, 3.0], &[0.0, 2.0, 1.0, 4.0, 3.0]);
        for (pos, &full) in cls.d_map.iter().enumerate() {
            assert_eq!(cls.full_to_d[full as usize], pos as Index);
        }
        for (pos, &full) in cls.c_map.iter().enumerate() {
            assert_eq!(cls.full_to_c[full as usize], pos as Index);
        }
    }
}
