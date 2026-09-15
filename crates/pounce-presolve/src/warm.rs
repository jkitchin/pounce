//! Warm-start-aware presolve, retain the transformation, map warm points.
//!
//! Seeds live in original space but the solver sees the reduced space
//! (tightened bounds, dropped rows, aux-clamped variables). This module
//! bridges the two: [`PresolveMap`] snapshots the transformation,
//! [`PresolveFingerprint`] detects when it went stale, and
//! [`project_warm_point`] maps a seed into reduced space.
//!
//! Dropped-row duals are dropped and reported
//! ([`WarmProjectionReport::dropped_dual_l1`]), not lost: redundant rows
//! carry 0 at the optimum and aux-eliminated rows are re-derived from KKT
//! stationarity at postsolve.
//!
//! The fingerprint does not see FBBT tapes (they arrive via
//! `ExpressionProvider`) or nonlinear data read only by auxiliary Phase 0 —
//! call [`PresolveTnlp::invalidate`](crate::PresolveTnlp::invalidate) after
//! changing those.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{BoundsInfo, IndexStyle, Linearity, SparsityRequest, TNLP};

use crate::linear_eq_plan::EliminationPlan;
use crate::options::PresolveOptions;

// FNV-1a over bit patterns: change detection, not a hash table.
fn mix(h: u64, v: u64) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    (h ^ v).wrapping_mul(PRIME)
}

fn hash_usize(h: u64, v: usize) -> u64 {
    mix(h, v as u64)
}

fn hash_f64(h: u64, v: Number) -> u64 {
    // -0.0 and 0.0 hash alike; NaNs hash by payload.
    let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
    mix(h, bits)
}

/// Clamp `v` into `[lo, hi]`, returning the clamped value, or `None`
/// when there is nothing valid to do: `v` already inside, any of
/// `v`/`lo`/`hi` NaN, or the box inverted (`lo > hi`).
pub fn clamp_seed(v: Number, lo: Number, hi: Number) -> Option<Number> {
    if v.is_nan() || lo.is_nan() || hi.is_nan() || lo > hi {
        return None;
    }
    let c = v.clamp(lo, hi);
    if c == v { None } else { Some(c) }
}

// Original-space seed (`x`/`z_l`/`z_u` length `n`, `lambda` length `m`).
// `mu` overrides the barrier seed; `None` threads the previous `final_mu`.
#[derive(Debug, Clone, Default)]
pub struct WarmPoint {
    /// Primal guess (length `n`).
    pub x: Vec<Number>,
    /// Constraint multipliers (length `m`, original row order).
    pub lambda: Vec<Number>,
    /// Lower-bound multipliers (length `n`).
    pub z_l: Vec<Number>,
    /// Upper-bound multipliers (length `n`).
    pub z_u: Vec<Number>,
    /// Barrier seed override. `None` threads the previous `final_mu`.
    pub mu: Option<Number>,
}

impl WarmPoint {
    /// True when every vector has the shape of an `(n, m)` problem.
    pub fn is_shaped(&self, n: usize, m: usize) -> bool {
        self.x.len() == n && self.z_l.len() == n && self.z_u.len() == n && self.lambda.len() == m
    }
}

// Retained transformation snapshot (see `PresolveTnlp::transformation`).

/// Everything needed to map an original-space warm point into reduced space
/// without holding the wrapper borrowed.
#[derive(Debug, Clone, Default)]
pub struct PresolveMap {
    /// Original variable count (unchanged by Phases 0–5).
    pub n_inner: usize,
    /// Original constraint count.
    pub m_inner: usize,
    /// Reduced constraint count (`rows_kept.len()`).
    pub m_outer: usize,
    /// `rows_kept[outer] = inner`: kept-row index map.
    pub rows_kept: Vec<usize>,
    /// Tightened variable lower bounds (reduced box).
    pub x_l: Vec<Number>,
    /// Tightened variable upper bounds (reduced box).
    pub x_u: Vec<Number>,
    /// Aux-fixed variable indices (union over the reduction stack).
    pub fixed_vars: Vec<usize>,
    /// Their solved values, aligned with `fixed_vars`.
    pub fixed_values: Vec<Number>,
}

impl PresolveMap {
    /// Identity map: presolve disabled or changed nothing observable.
    pub fn identity(n: usize, m: usize, x_l: Vec<Number>, x_u: Vec<Number>) -> Self {
        Self {
            n_inner: n,
            m_inner: m,
            m_outer: m,
            rows_kept: (0..m).collect(),
            x_l,
            x_u,
            fixed_vars: Vec::new(),
            fixed_values: Vec::new(),
        }
    }

    /// True for [`Self::identity`]-shaped maps with no fixed variables.
    pub fn is_identity(&self) -> bool {
        self.fixed_vars.is_empty()
            && self.m_inner == self.m_outer
            && self.rows_kept.iter().enumerate().all(|(o, &i)| o == i)
    }
}

// What projection did to a warm point.

/// Counts for callers tuning a sweep or asserting the warm path engages.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WarmProjectionReport {
    /// Reduced problem's missing rows (warm dual mass dropped).
    pub n_dropped_rows: usize,
    /// `Σ|λ|` over dropped rows.
    pub dropped_dual_l1: Number,
    /// Warm primals clamped into the tightened box.
    pub x_clamped_count: usize,
    /// Warm primals overridden with aux-fixed values.
    pub x_fixed_overridden_count: usize,
}

// Reduced-space seed: what the solver consumes.
#[derive(Debug, Clone)]
pub struct ProjectedWarm {
    /// Primal in reduced space (length `n_inner` from
    /// [`project_warm_point`]; shorter after
    /// [`project_warm_point_full`] gathers elimination survivors).
    pub x: Vec<Number>,
    /// Multipliers on kept rows (length `m_outer`).
    pub lambda: Vec<Number>,
    /// Lower-bound multipliers (length `n_inner`).
    pub z_l: Vec<Number>,
    /// Upper-bound multipliers (length `n_inner`).
    pub z_u: Vec<Number>,
    /// What the projection did.
    pub report: WarmProjectionReport,
}

/// Map `warm` (original space) through `map` into reduced space.
///
/// Returns `None` on a shape mismatch — the caller falls back to cold.
/// This is the presolve layer only; when a linear-eq elimination is
/// stacked outside presolve, use [`project_warm_point_full`] so the
/// report describes the whole chain the solver consumes.
pub fn project_warm_point(map: &PresolveMap, warm: &WarmPoint) -> Option<ProjectedWarm> {
    if !warm.is_shaped(map.n_inner, map.m_inner) {
        return None;
    }
    let n = map.n_inner;
    let mut x = warm.x.clone();
    let mut report = WarmProjectionReport {
        n_dropped_rows: map.m_inner.saturating_sub(map.m_outer),
        ..WarmProjectionReport::default()
    };
    // Aux-fixed variables are clamped in the reduced problem; the warm
    // primal there is stale, so override it (counting actual moves).
    for (k, &i) in map.fixed_vars.iter().enumerate() {
        if i < n && k < map.fixed_values.len() {
            if x[i] != map.fixed_values[k] {
                report.x_fixed_overridden_count += 1;
            }
            x[i] = map.fixed_values[k];
        }
    }
    // Clamp into the tightened box.
    for (i, v) in x.iter_mut().enumerate().take(n) {
        if i < map.x_l.len() && i < map.x_u.len() {
            if let Some(c) = clamp_seed(*v, map.x_l[i], map.x_u[i]) {
                report.x_clamped_count += 1;
                *v = c;
            }
        }
    }
    // Mask multipliers to kept rows; report dropped mass.
    let mut lambda = vec![0.0; map.m_outer];
    for (outer, &inner) in map.rows_kept.iter().enumerate() {
        if inner < warm.lambda.len() && outer < lambda.len() {
            lambda[outer] = warm.lambda[inner];
        }
    }
    if map.m_outer < map.m_inner {
        let mut kept = vec![false; map.m_inner];
        for &i in &map.rows_kept {
            if i < kept.len() {
                kept[i] = true;
            }
        }
        report.dropped_dual_l1 = warm
            .lambda
            .iter()
            .enumerate()
            .filter(|(i, _)| !kept[*i])
            .map(|(_, &v)| v.abs())
            .sum();
    }
    Some(ProjectedWarm {
        x,
        lambda,
        z_l: warm.z_l.clone(),
        z_u: warm.z_u.clone(),
        report,
    })
}

/// Map `warm` through the presolve layer and the linear-eq
/// elimination stacked outside it, mirroring what the solver
/// consumes.
pub fn project_warm_point_full(
    map: &PresolveMap,
    elim: Option<&EliminationPlan>,
    warm: &WarmPoint,
) -> Option<ProjectedWarm> {
    let mut proj = project_warm_point(map, warm)?;
    let plan = match elim {
        None => return Some(proj),
        Some(p) => p,
    };
    if proj.x.len() != plan.n_full || proj.lambda.len() != plan.m_full {
        return None;
    }
    let mut x = vec![0.0; plan.vars_kept.len()];
    let mut z_l = vec![0.0; plan.vars_kept.len()];
    let mut z_u = vec![0.0; plan.vars_kept.len()];
    for (red, &full) in plan.vars_kept.iter().enumerate() {
        if full < proj.x.len() && red < x.len() {
            x[red] = proj.x[full];
            z_l[red] = proj.z_l[full];
            z_u[red] = proj.z_u[full];
        }
    }
    let mut lambda = vec![0.0; plan.rows_kept.len()];
    for (red, &full) in plan.rows_kept.iter().enumerate() {
        if full < proj.lambda.len() && red < lambda.len() {
            lambda[red] = proj.lambda[full];
        }
    }
    let mut kept = vec![false; plan.m_full];
    for &f in &plan.rows_kept {
        if f < kept.len() {
            kept[f] = true;
        }
    }
    for (i, &is_kept) in kept.iter().enumerate() {
        if !is_kept {
            proj.report.n_dropped_rows += 1;
            proj.report.dropped_dual_l1 += proj.lambda[i].abs();
        }
    }
    proj.x = x;
    proj.lambda = lambda;
    proj.z_l = z_l;
    proj.z_u = z_u;
    Some(proj)
}

// What the transformation was computed from. The session rebuilds the
// wrapper on any change; bounds and linear-row values are hashed because
// tightening and redundancy read them.

/// Recomputed from the live TNLP before each solve; a match reuses the
/// retained wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresolveFingerprint {
    /// Combined hash; compared as a whole.
    hash: u64,
}

impl PresolveFingerprint {
    /// The combined hash (for logging/diagnostics).
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

/// Compute the fingerprint of `inner` under `opts`.
///
/// `None` on any failed query means "always rebuild". A declined
/// Hessian-structure call just records absence.
pub fn compute_fingerprint(
    inner: &Rc<RefCell<dyn TNLP>>,
    opts: &PresolveOptions,
) -> Option<PresolveFingerprint> {
    const SEED: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h = SEED;

    let info = inner.borrow_mut().get_nlp_info()?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    let nnz = info.nnz_jac_g.max(0) as usize;
    let nnz_h = info.nnz_h_lag.max(0) as usize;
    h = hash_usize(h, n);
    h = hash_usize(h, m);
    h = hash_usize(h, nnz);
    h = hash_usize(h, nnz_h);
    h = hash_usize(h, info.index_style as usize);

    // Bounds: tightening/redundancy read them.
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !inner.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }
    for v in x_l
        .iter()
        .chain(x_u.iter())
        .chain(g_l.iter())
        .chain(g_u.iter())
    {
        h = hash_f64(h, *v);
    }

    // Linearity tags gate every phase's row eligibility.
    let mut lin = vec![Linearity::NonLinear; m];
    let have_lin = if m > 0 {
        inner.borrow_mut().get_constraints_linearity(&mut lin)
    } else {
        true
    };
    if m > 0 {
        if have_lin {
            for t in &lin {
                h = hash_usize(
                    h,
                    match t {
                        Linearity::Linear => 1,
                        Linearity::NonLinear => 2,
                    },
                );
            }
        } else {
            h = hash_usize(h, 0);
        }
    }

    if n > 0 {
        let mut var_lin = vec![Linearity::NonLinear; n];
        let have_var_lin = {
            let mut inner = inner.borrow_mut();
            inner.get_objective_variables_linearity(&mut var_lin)
                || inner.get_variables_linearity(&mut var_lin)
        };
        h = hash_usize(h, have_var_lin as usize);
        if have_var_lin {
            for t in &var_lin {
                h = hash_usize(
                    h,
                    match t {
                        Linearity::Linear => 1,
                        Linearity::NonLinear => 2,
                    },
                );
            }
        }
    }

    // Jacobian structure + values at the probe. Linear rows are constant,
    // so any accepted probe gives the exact coefficients presolve uses.
    let mut irow = vec![0 as Index; nnz];
    let mut jcol = vec![0 as Index; nnz];
    if nnz > 0 {
        if !inner.borrow_mut().eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) {
            return None;
        }
        for v in irow.iter().chain(jcol.iter()) {
            h = mix(h, *v as u64);
        }
    }
    // Probe point for values.
    let mut x_probe = vec![0.0; n];
    let mut zl_probe = vec![0.0; n];
    let mut zu_probe = vec![0.0; n];
    let mut lam_probe = vec![0.0; m];
    if !inner
        .borrow_mut()
        .get_starting_point(pounce_nlp::tnlp::StartingPoint {
            init_x: true,
            x: &mut x_probe,
            init_z: false,
            z_l: &mut zl_probe,
            z_u: &mut zu_probe,
            init_lambda: false,
            lambda: &mut lam_probe,
        })
    {
        return None;
    }
    if nnz > 0 {
        let mut values = vec![0.0; nnz];
        if !inner.borrow_mut().eval_jac_g(
            Some(&x_probe),
            true,
            SparsityRequest::Values {
                values: &mut values,
            },
        ) {
            return None;
        }
        let base = match info.index_style {
            IndexStyle::C => 0 as Index,
            IndexStyle::Fortran => 1 as Index,
        };
        for (k, v) in values.iter().enumerate() {
            // Row of entry `k`, converted out of the TNLP's index style.
            let row = if k < irow.len() {
                irow[k] - base
            } else {
                -1 as Index
            };
            let linear = have_lin
                && row >= 0
                && (row as usize) < m
                && lin[row as usize] == Linearity::Linear;
            if linear || !have_lin || row < 0 {
                h = hash_f64(h, *v);
            }
        }
    }

    // Hessian structure presence: the elimination pass keys shape off it.
    if nnz_h > 0 {
        let mut hi = vec![0 as Index; nnz_h];
        let mut hj = vec![0 as Index; nnz_h];
        let ok = inner.borrow_mut().eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut hi,
                jcol: &mut hj,
            },
        );
        h = hash_usize(h, if ok { 1000 + nnz_h } else { 999 });
        if ok {
            for v in hi.iter().chain(hj.iter()) {
                h = mix(h, *v as u64);
            }
        }
    }

    // Options: any knob change rebuilds.
    h = hash_presolve_options(h, opts);

    Some(PresolveFingerprint { hash: h })
}

fn hash_presolve_options(mut h: u64, opts: &PresolveOptions) -> u64 {
    h = hash_usize(h, opts.enabled as usize);
    h = hash_f64(h, opts.certify_tol);
    h = hash_usize(h, opts.bound_tightening as usize);
    h = hash_usize(h, opts.redundant_constraint_removal as usize);
    h = hash_usize(h, opts.linear_eq_reduction as usize);
    h = hash_usize(h, opts.licq_check as usize);
    h = hash_usize(h, opts.print_level as usize);
    h = hash_usize(h, opts.max_passes as usize);
    h = hash_usize(
        h,
        match opts.licq_action {
            crate::options::LicqAction::Warn => 1,
            crate::options::LicqAction::AutoL1 => 2,
        },
    );
    h = hash_usize(h, opts.warm_z_bounds as usize);
    h = hash_f64(h, opts.bound_mult_init_val);
    h = hash_usize(h, opts.auxiliary as usize);
    h = hash_f64(h, opts.auxiliary_tol);
    h = hash_usize(h, opts.auxiliary_max_block_dim as usize);
    h = hash_f64(h, opts.auxiliary_wall_time_fraction);
    h = hash_usize(
        h,
        match opts.auxiliary_coupling {
            crate::options::AuxiliaryCouplingPolicy::None => 1,
            crate::options::AuxiliaryCouplingPolicy::Safe => 2,
            crate::options::AuxiliaryCouplingPolicy::Aggressive => 3,
        },
    );
    h = hash_usize(h, opts.auxiliary_diagnostics as usize);
    h = hash_usize(h, opts.fbbt as usize);
    h = hash_f64(h, opts.fbbt_tol);
    h = hash_usize(h, opts.fbbt_max_iter as usize);
    h = hash_usize(h, opts.fbbt_max_constraints as usize);
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::tnlp::{
        BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest,
        StartingPoint,
    };

    /// min x^2 s.t. x >= 1 and (redundant) x <= 100.
    struct Mini {
        x_l: f64,
        x_u: f64,
    }

    impl TNLP for Mini {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 2,
                nnz_jac_g: 2,
                nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = self.x_l;
            b.x_u[0] = self.x_u;
            b.g_l[0] = 1.0;
            b.g_u[0] = 1e19;
            b.g_l[1] = -1e19;
            b.g_u[1] = 100.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 0.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
            Some(x[0] * x[0])
        }
        fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * x[0];
            true
        }
        fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = x[0];
            g[1] = x[0];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 1]);
                    jcol.copy_from_slice(&[0, 0]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
            types.fill(Linearity::Linear);
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn mini() -> Rc<RefCell<dyn TNLP>> {
        Rc::new(RefCell::new(Mini {
            x_l: -10.0,
            x_u: 10.0,
        })) as Rc<RefCell<dyn TNLP>>
    }

    #[test]
    fn projection_masks_dropped_lambda_and_reports_mass() {
        let map = PresolveMap {
            n_inner: 1,
            m_inner: 2,
            m_outer: 1,
            rows_kept: vec![0],
            x_l: vec![1.0],
            x_u: vec![10.0],
            fixed_vars: vec![],
            fixed_values: vec![],
        };
        let warm = WarmPoint {
            x: vec![0.0], // below tightened box -> clamped
            lambda: vec![2.0, 5.0],
            z_l: vec![0.5],
            z_u: vec![0.0],
            mu: None,
        };
        let proj = project_warm_point(&map, &warm).expect("shaped");
        assert_eq!(proj.x, vec![1.0]);
        assert_eq!(proj.lambda, vec![2.0]);
        assert_eq!(proj.report.n_dropped_rows, 1);
        assert!((proj.report.dropped_dual_l1 - 5.0).abs() < 1e-12);
        assert_eq!(proj.report.x_clamped_count, 1);
        assert_eq!(proj.report.x_fixed_overridden_count, 0);
    }

    #[test]
    fn projection_overrides_fixed_vars() {
        let map = PresolveMap {
            n_inner: 2,
            m_inner: 1,
            m_outer: 1,
            rows_kept: vec![0],
            x_l: vec![-10.0, 3.0],
            x_u: vec![10.0, 3.0],
            fixed_vars: vec![1],
            fixed_values: vec![3.0],
        };
        let warm = WarmPoint {
            x: vec![1.0, 99.0],
            lambda: vec![1.0],
            z_l: vec![0.0, 0.0],
            z_u: vec![0.0, 0.0],
            mu: None,
        };
        let proj = project_warm_point(&map, &warm).expect("shaped");
        assert_eq!(proj.x, vec![1.0, 3.0]);
        assert_eq!(proj.report.x_fixed_overridden_count, 1);
    }

    #[test]
    fn projection_rejects_misshaped_warm() {
        let map = PresolveMap::identity(2, 1, vec![-10.0, -10.0], vec![10.0, 10.0]);
        let warm = WarmPoint {
            x: vec![0.0], // wrong n
            lambda: vec![0.0],
            z_l: vec![0.0, 0.0],
            z_u: vec![0.0, 0.0],
            mu: None,
        };
        assert!(project_warm_point(&map, &warm).is_none());
    }

    #[test]
    fn fingerprint_stable_then_moves_with_bounds() {
        let inner = mini();
        let opts = PresolveOptions::defaults();
        let a = compute_fingerprint(&inner, &opts).expect("fingerprint");
        let b = compute_fingerprint(&inner, &opts).expect("fingerprint");
        assert_eq!(a, b);
        let inner2: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mini {
            x_l: -5.0,
            x_u: 10.0,
        }));
        let c = compute_fingerprint(&inner2, &opts).expect("fingerprint");
        assert_ne!(a, c);
    }

    #[test]
    fn fingerprint_moves_with_options() {
        let inner = mini();
        let mut opts = PresolveOptions::defaults();
        let a = compute_fingerprint(&inner, &opts).expect("fingerprint");
        opts.redundant_constraint_removal = false;
        let b = compute_fingerprint(&inner, &opts).expect("fingerprint");
        assert_ne!(a, b);
    }

    /// One linear row (`lin_coef * x == 1`), one nonlinear row (`x^2 <= h`)
    /// whose Jacobian value moves with the starting point.
    struct NonlinMini {
        x0: f64,
        lin_coef: f64,
    }

    impl TNLP for NonlinMini {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 2,
                nnz_jac_g: 2,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = -10.0;
            b.x_u[0] = 10.0;
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            b.g_l[1] = -1e19;
            b.g_u[1] = 100.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = self.x0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
            Some(x[0] * x[0])
        }
        fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * x[0];
            true
        }
        fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = self.lin_coef * x[0];
            g[1] = x[0] * x[0];
            true
        }
        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            _n: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 1]);
                    jcol.copy_from_slice(&[0, 0]);
                }
                SparsityRequest::Values { values } => {
                    let at = x.map(|v| v[0]).unwrap_or(self.x0);
                    values.copy_from_slice(&[self.lin_coef, 2.0 * at]);
                }
            }
            true
        }
        fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
            types[0] = Linearity::Linear;
            types[1] = Linearity::NonLinear;
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn fingerprint_ignores_nonlinear_operating_point() {
        let opts = PresolveOptions::defaults();
        let mk = |x0: f64, lin_coef: f64| -> Rc<RefCell<dyn TNLP>> {
            Rc::new(RefCell::new(NonlinMini { x0, lin_coef })) as Rc<RefCell<dyn TNLP>>
        };
        let a = compute_fingerprint(&mk(0.0, 1.0), &opts).expect("fingerprint");
        // Same linear data, new operating point (nonlinear Jacobian value
        // 0.0 -> 10.0): must reuse, not rebuild.
        let b = compute_fingerprint(&mk(5.0, 1.0), &opts).expect("fingerprint");
        assert_eq!(a, b, "nonlinear operating point must not rebuild");
        // Same operating point, moved linear coefficient: must rebuild.
        let c = compute_fingerprint(&mk(0.0, 2.0), &opts).expect("fingerprint");
        assert_ne!(a, c, "linear coefficient change must rebuild");
    }

    #[test]
    fn full_projection_folds_elim_gather() {
        use crate::linear_eq_plan::EliminationPlan;

        let map = PresolveMap {
            n_inner: 2,
            m_inner: 2,
            m_outer: 2,
            rows_kept: vec![0, 1],
            x_l: vec![-10.0, -10.0],
            x_u: vec![10.0, 10.0],
            fixed_vars: vec![],
            fixed_values: vec![],
        };
        let warm = WarmPoint {
            x: vec![1.0, 2.0],
            lambda: vec![3.0, 4.0],
            z_l: vec![0.5, 0.25],
            z_u: vec![0.0, 0.0],
            mu: None,
        };
        let mut plan = EliminationPlan::identity(2, 2, &[-10.0, -10.0], &[10.0, 10.0]);
        plan.vars_kept = vec![1];
        plan.rows_kept = vec![1];
        let proj = project_warm_point_full(&map, Some(&plan), &warm).expect("shaped");
        assert_eq!(proj.x, vec![2.0]);
        assert_eq!(proj.lambda, vec![4.0]);
        assert_eq!(proj.z_l, vec![0.25]);
        assert_eq!(proj.report.n_dropped_rows, 1);
        assert!((proj.report.dropped_dual_l1 - 3.0).abs() < 1e-12);
        let single = project_warm_point(&map, &warm).expect("shaped");
        let passthrough = project_warm_point_full(&map, None, &warm).expect("shaped");
        assert_eq!(passthrough.x, single.x);
        assert_eq!(passthrough.lambda, single.lambda);
        assert_eq!(passthrough.report, single.report);
    }

    #[test]
    fn full_projection_rejects_elim_shape_mismatch() {
        use crate::linear_eq_plan::EliminationPlan;

        let map = PresolveMap::identity(2, 1, vec![-10.0, -10.0], vec![10.0, 10.0]);
        let warm = WarmPoint {
            x: vec![0.0, 0.0],
            lambda: vec![0.0],
            z_l: vec![0.0, 0.0],
            z_u: vec![0.0, 0.0],
            mu: None,
        };
        let plan = EliminationPlan::identity(3, 1, &[-10.0, -10.0, -10.0], &[10.0, 10.0, 10.0]);
        assert!(project_warm_point_full(&map, Some(&plan), &warm).is_none());
    }
}
