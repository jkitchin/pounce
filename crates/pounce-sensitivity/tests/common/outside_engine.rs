//! An interior-point engine outside pounce, as the corrector and the activity
//! classification see one.
//!
//! [`OutsideEngine`] implements `SensBacksolver` and `BarrierIterate` from
//! nothing pounce computed internally: a model's own callbacks, a final
//! iterate as an engine reports it to its caller (primal point, constraint
//! multipliers, variable-bound multipliers), the barrier parameter it stopped
//! at, and the bounds it iterated against. It fills in what such a report
//! leaves out, the inequality slacks and their bound multipliers, and it
//! assembles and factors its own KKT matrix with a dense LU.
//!
//! The KKT matrix is the full eight-block Newton matrix of the barrier
//! residual, with the bound-multiplier rows kept rather than eliminated:
//!
//! ```text
//!        x      s     y_c    y_d    z_l    z_u    v_l    v_u
//! x    [ W      .     J_cᵀ   J_dᵀ   −P_xl  P_xu   .      .    ]
//! s    [ .      .     .      −I     .      .      −P_dl  P_du ]
//! y_c  [ J_c    .     .      .      .      .      .      .    ]
//! y_d  [ J_d    −I    .      .      .      .      .      .    ]
//! z_l  [ Z_l Pᵀ .     .      .      S_xl   .      .      .    ]
//! z_u  [−Z_u Pᵀ .     .      .      .      S_xu   .      .    ]
//! v_l  [ .   V_l Pᵀ   .      .      .      .      S_dl   .    ]
//! v_u  [ .  −V_u Pᵀ   .      .      .      .      .      S_du ]
//! ```
//!
//! Eliminating the last four block rows gives the augmented system with
//! `Σ = Z/S` on the diagonal, which is what pounce factors, so the two
//! agree in exact arithmetic. Everything here is in natural units, so the
//! engine frame and the natural frame are one.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linalg::{Matrix, SymMatrix};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, InfeasibilityProof, IpoptCq, IpoptData, IterStats, Linearity, MetaData,
    NlpInfo, ScalingRequest, Solution, SparsityRequest, StartingPoint,
};
use pounce_nlp::{DerivativeProofs, TNLP};
use pounce_sens_core::backsolver::{BoundRow, DenseLuBacksolver, SensBacksolver};
use pounce_sens_core::barrier::{
    ActivityData, BarrierIterate, BoundSide, EngineScaling, KktOperator, ModelMaps,
};
use pounce_sens_core::schur_data::IndexSchurData;
use pounce_sens_core::sens_app::{SensApplication, SensOptions};
use pounce_sensitivity::solver::BARRIER_SIGN;

/// The infinity convention of Ipopt and pounce: a bound at or beyond this
/// magnitude does not exist.
const BOUND_INF: Number = 1e19;

// ---------------------------------------------------------------------------
// Recording what pounce returns
// ---------------------------------------------------------------------------

/// A final iterate as an engine reports it to its caller.
#[derive(Debug, Clone, Default)]
pub struct Reported {
    pub x: Vec<Number>,
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
    pub lambda: Vec<Number>,
    /// The barrier parameter of the last iteration, from the iteration
    /// callback.
    pub mu: Number,
}

/// Forwards every `TNLP` method to `inner`, and records the solution
/// `finalize_solution` receives and the barrier parameter the last
/// `intermediate_callback` reports.
pub struct RecordingTnlp<T: TNLP> {
    pub inner: T,
    pub reported: Option<Reported>,
    last_mu: Number,
}

impl<T: TNLP> RecordingTnlp<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            reported: None,
            last_mu: Number::NAN,
        }
    }
}

impl<T: TNLP> TNLP for RecordingTnlp<T> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.eval_f(x, new_x)
    }
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.eval_grad_f(x, new_x, grad_f)
    }
    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.eval_g(x, new_x, g)
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.eval_jac_g(x, new_x, mode)
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        self.reported = Some(Reported {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            lambda: sol.lambda.to_vec(),
            mu: self.last_mu,
        });
        self.inner.finalize_solution(sol, ip_data, ip_cq);
    }
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.get_var_con_metadata(var, con)
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.get_scaling_parameters(req)
    }
    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_variables_linearity(types)
    }
    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_objective_variables_linearity(types)
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.get_constraints_linearity(types)
    }
    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.get_number_of_nonlinear_variables()
    }
    fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
        self.inner.get_list_of_nonlinear_variables(pos)
    }
    fn derivative_proofs(&mut self) -> DerivativeProofs {
        self.inner.derivative_proofs()
    }
    /// Always: the barrier parameter of the last iteration is read here.
    /// The callback only observes, and the inner model's own callback runs
    /// only when it asked for one.
    fn wants_intermediate_callback(&self) -> bool {
        true
    }
    fn intermediate_callback(&mut self, stats: IterStats, d: &IpoptData, q: &IpoptCq) -> bool {
        self.last_mu = stats.mu;
        if self.inner.wants_intermediate_callback() {
            self.inner.intermediate_callback(stats, d, q)
        } else {
            true
        }
    }
    fn wants_current_iterate(&self) -> bool {
        self.inner.wants_current_iterate()
    }
    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.finalize_metadata(var, con)
    }
    fn is_presolve_wrapper(&self) -> bool {
        self.inner.is_presolve_wrapper()
    }
    fn scaling_factors(&self) -> Option<Vec<Number>> {
        self.inner.scaling_factors()
    }
    fn presolve_infeasibility_proof(&self) -> Option<InfeasibilityProof> {
        self.inner.presolve_infeasibility_proof()
    }
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

/// Variable and row bounds, in the model's indexing.
#[derive(Debug, Clone)]
pub struct Bounds {
    pub x_l: Vec<Number>,
    pub x_u: Vec<Number>,
    pub g_l: Vec<Number>,
    pub g_u: Vec<Number>,
}

impl Bounds {
    /// The bounds the model declares.
    pub fn declared(model: &mut dyn TNLP) -> Self {
        let info = model.get_nlp_info().expect("nlp info");
        let (n, m) = (info.n as usize, info.m as usize);
        let mut b = Bounds {
            x_l: vec![0.0; n],
            x_u: vec![0.0; n],
            g_l: vec![0.0; m],
            g_u: vec![0.0; m],
        };
        assert!(model.get_bounds_info(BoundsInfo {
            x_l: &mut b.x_l,
            x_u: &mut b.x_u,
            g_l: &mut b.g_l,
            g_u: &mut b.g_u,
        }));
        b
    }

    /// The bounds pounce iterates against under `bound_relax_factor`:
    /// a variable bound moves out by `min(factor·max(1, |b|), cap)`, and an
    /// inequality-row bound by `min(factor, cap)·|b|`, or by
    /// `min(factor, cap)` when `b` is zero. Fixed variables and equality
    /// rows do not move. `cap` is `constr_viol_tol`.
    pub fn relaxed_as_pounce(&self, factor: Number, cap: Number) -> Self {
        if factor <= 0.0 {
            return self.clone();
        }
        let mut out = self.clone();
        for i in 0..self.x_l.len() {
            if self.x_l[i] == self.x_u[i] {
                continue;
            }
            out.x_l[i] -= (factor * self.x_l[i].abs().max(1.0)).min(cap);
            out.x_u[i] += (factor * self.x_u[i].abs().max(1.0)).min(cap);
        }
        let w = factor.min(cap);
        let scale = |b: Number| if b == 0.0 { 1.0 } else { b.abs() };
        for j in 0..self.g_l.len() {
            if self.g_l[j] == self.g_u[j] {
                continue;
            }
            out.g_l[j] -= w * scale(self.g_l[j]);
            out.g_u[j] += w * scale(self.g_u[j]);
        }
        out
    }

    /// The bounds Ipopt iterates against under `bound_relax_factor`: every
    /// bound moves out by `min(factor·max(1, |b|), cap)`.
    pub fn relaxed_as_ipopt(&self, factor: Number, cap: Number) -> Self {
        let mut out = self.clone();
        let d = |b: Number| (factor * b.abs().max(1.0)).min(cap);
        for i in 0..self.x_l.len() {
            if self.x_l[i] == self.x_u[i] {
                continue;
            }
            out.x_l[i] -= d(self.x_l[i]);
            out.x_u[i] += d(self.x_u[i]);
        }
        for j in 0..self.g_l.len() {
            if self.g_l[j] == self.g_u[j] {
                continue;
            }
            out.g_l[j] -= d(self.g_l[j]);
            out.g_u[j] += d(self.g_u[j]);
        }
        out
    }
}

/// The slack an Ipopt-family engine divides by: the raw distance to the
/// bound, raised where it is below `ε·min(1, μ)` to `μ/z`, but by no more
/// than `slack_move·max(1, |bound|)`, with `slack_move = ε^(3/4)`.
///
/// This is Ipopt's `CalculateSafeSlack`, which pounce ports. It changes
/// nothing but a slack at or within roundoff of zero, the case of a point
/// parked exactly on its bound, where the raw slack would put an infinite
/// barrier diagonal in the KKT matrix.
fn safe_slack(raw: Number, bound: Number, z: Number, mu: Number) -> Number {
    let mut s_min = Number::EPSILON * mu.min(1.0);
    if s_min == 0.0 {
        s_min = Number::MIN_POSITIVE;
    }
    if raw >= s_min {
        return raw;
    }
    let slack = raw.max(0.0);
    let target = if mu != 0.0 {
        (mu / z).max(s_min)
    } else {
        s_min
    };
    let slack_move = Number::EPSILON.powf(0.75);
    target
        .min(slack + slack_move * bound.abs().max(1.0))
        .max(s_min)
}

fn finite_lower(b: Number) -> bool {
    b > -BOUND_INF
}

fn finite_upper(b: Number) -> bool {
    b < BOUND_INF
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// An engine's converged iterate in the compound layout, with its own model
/// evaluations and its own factorization.
#[derive(Clone)]
pub struct OutsideEngine {
    model: Rc<RefCell<dyn TNLP>>,
    index_base: Index,
    n_full: usize,
    /// The model's index of each variable kept, in order. A variable whose
    /// bounds coincide is removed, as both engines do by default.
    free: Vec<usize>,
    /// The full primal point with the removed variables at their values,
    /// overwritten with a compound point's `x` block before each evaluation.
    x_template: Vec<Number>,
    /// The model's equality rows and their right-hand sides.
    c_rows: Vec<usize>,
    c_rhs: Vec<Number>,
    /// The model's inequality rows.
    d_rows: Vec<usize>,
    /// For each model row, its position among the inequality rows.
    full_g_to_d: Vec<Option<usize>>,
    /// Whether the reported point is a vertex crossover parked on its
    /// bounds, which changes how the classification reads its diagonal.
    crossover: bool,
    /// The declared bounds, over the kept variables and the inequality rows.
    x_decl_lo: Vec<Number>,
    x_decl_hi: Vec<Number>,
    d_decl_lo: Vec<Number>,
    d_decl_hi: Vec<Number>,
    /// The relaxed bounds, over the kept variables and the inequality rows.
    x_lo: Vec<Number>,
    x_hi: Vec<Number>,
    d_lo: Vec<Number>,
    d_hi: Vec<Number>,
    /// Which kept variable or inequality row each bound belongs to.
    xl_pos: Vec<usize>,
    xu_pos: Vec<usize>,
    dl_pos: Vec<usize>,
    du_pos: Vec<usize>,
    dims: [usize; 8],
    off: [usize; 9],
    bound_rows: Vec<BoundRow>,
    jac_structure: Vec<(usize, usize)>,
    hess_structure: Vec<(usize, usize)>,
    /// The factor the engine multiplied the objective by, `-1` for a
    /// maximization it turned into a minimization. The compound iterate
    /// holds the multipliers of that minimization, so they are non-negative
    /// at every bound.
    obj_factor: Number,
    /// The pair that carries the engine frame to natural units when the
    /// objective was scaled: `E` multiplies a natural right-hand side into
    /// the engine frame and `F` a solution out of it, so `K⁻¹ = F·K̃⁻¹·E`.
    /// For an objective factor `σ` and no other scaling, `E` is `σ` on the
    /// `x`, `s`, `z` and `v` rows and one on the multiplier rows, and `F` is
    /// one on the primal rows and `1/σ` on every multiplier row. `None` when
    /// `σ = 1`, which is the identity.
    conj: Option<(Vec<Number>, Vec<Number>)>,
    /// The converged compound iterate.
    base: Vec<Number>,
    /// The barrier level the converged point sits on, what the analyses read.
    mu: Number,
    /// The engine's own barrier parameter at its last iteration, what its
    /// slack safeguard reads. The two differ only on a point whose bound
    /// multipliers a terminating path set to zero.
    slack_mu: Number,
    base_lu: Factored,
    /// What the engine reports wrong on purpose, `None` for nothing.
    fault: Option<Fault>,
}

/// The order the engine keeps its variables and rows in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// The model's order, which is pounce's: its blocks and the model's
    /// indices then agree position by position, so a calculation that read
    /// a block position as a model index would still be right.
    ModelOrder,
    /// Every block in the reverse of the model's order. Only the maps the
    /// engine reports relate the two, so an answer in model indices that
    /// moves with the layout is a map read wrong.
    Reversed,
}

/// A report the engine gets wrong on purpose, so a test can check that the
/// calculations check what an engine reports rather than trusting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The variable map names its first variable twice.
    Maps,
    /// The equality rows' scale factors are one short.
    Scaling,
    /// The primal bounds are one short.
    PrimalBounds,
}

impl OutsideEngine {
    /// Builds the engine from a reported iterate.
    ///
    /// `declared` decides the structure, which variables are fixed and which
    /// rows are equalities, and `relaxed` the bounds the slacks are measured
    /// against.
    pub fn new(
        model: Rc<RefCell<dyn TNLP>>,
        reported: &Reported,
        declared: &Bounds,
        relaxed: &Bounds,
    ) -> Self {
        Self::with_objective_factor(model, reported, declared, relaxed, 1.0)
    }

    /// [`Self::new`] for an engine that minimized `obj_factor·f`.
    ///
    /// A report gives the multipliers in the model's own sense, the
    /// derivatives of `f` itself, so for a negative factor they come back
    /// with the opposite sign to the ones the engine iterated with. They are
    /// multiplied by `obj_factor` here to recover those.
    pub fn with_objective_factor(
        model: Rc<RefCell<dyn TNLP>>,
        reported: &Reported,
        declared: &Bounds,
        relaxed: &Bounds,
        obj_factor: Number,
    ) -> Self {
        Self::with_layout(
            model,
            reported,
            declared,
            relaxed,
            obj_factor,
            Layout::ModelOrder,
        )
    }

    /// [`Self::with_objective_factor`] with the engine's blocks in the
    /// order `layout` gives.
    pub fn with_layout(
        model: Rc<RefCell<dyn TNLP>>,
        reported: &Reported,
        declared: &Bounds,
        relaxed: &Bounds,
        obj_factor: Number,
        layout: Layout,
    ) -> Self {
        let mut reported = reported.clone();
        for v in reported
            .z_l
            .iter_mut()
            .chain(reported.z_u.iter_mut())
            .chain(reported.lambda.iter_mut())
        {
            *v *= obj_factor;
        }
        let reported = &reported;
        let (n_full, m, index_base, jac_structure, hess_structure) = {
            let mut md = model.borrow_mut();
            let info = md.get_nlp_info().expect("nlp info");
            let base = match info.index_style {
                IndexStyle::C => 0,
                IndexStyle::Fortran => 1,
            };
            let mut ir = vec![0; info.nnz_jac_g as usize];
            let mut jc = vec![0; info.nnz_jac_g as usize];
            assert!(md.eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut ir,
                    jcol: &mut jc,
                },
            ));
            let jac: Vec<(usize, usize)> = ir
                .iter()
                .zip(&jc)
                .map(|(&r, &c)| ((r - base) as usize, (c - base) as usize))
                .collect();
            let mut hr = vec![0; info.nnz_h_lag as usize];
            let mut hc = vec![0; info.nnz_h_lag as usize];
            let hess: Vec<(usize, usize)> = if info.nnz_h_lag > 0
                && md.eval_h(
                    None,
                    false,
                    1.0,
                    None,
                    false,
                    SparsityRequest::Structure {
                        irow: &mut hr,
                        jcol: &mut hc,
                    },
                ) {
                hr.iter()
                    .zip(&hc)
                    .map(|(&r, &c)| ((r - base) as usize, (c - base) as usize))
                    .collect()
            } else {
                Vec::new()
            };
            (info.n as usize, info.m as usize, base, jac, hess)
        };

        let mut free: Vec<usize> = (0..n_full)
            .filter(|&i| declared.x_l[i] != declared.x_u[i])
            .collect();
        let mut c_rows: Vec<usize> = (0..m)
            .filter(|&j| declared.g_l[j] == declared.g_u[j])
            .collect();
        let mut d_rows: Vec<usize> = (0..m)
            .filter(|&j| declared.g_l[j] != declared.g_u[j])
            .collect();
        if layout == Layout::Reversed {
            free.reverse();
            c_rows.reverse();
            d_rows.reverse();
        }
        let mut full_g_to_d = vec![None; m];
        for (k, &j) in d_rows.iter().enumerate() {
            full_g_to_d[j] = Some(k);
        }
        let c_rhs = c_rows.iter().map(|&j| declared.g_l[j]).collect();

        let x_decl_lo: Vec<Number> = free.iter().map(|&i| declared.x_l[i]).collect();
        let x_decl_hi: Vec<Number> = free.iter().map(|&i| declared.x_u[i]).collect();
        let d_decl_lo: Vec<Number> = d_rows.iter().map(|&j| declared.g_l[j]).collect();
        let d_decl_hi: Vec<Number> = d_rows.iter().map(|&j| declared.g_u[j]).collect();
        let x_lo: Vec<Number> = free.iter().map(|&i| relaxed.x_l[i]).collect();
        let x_hi: Vec<Number> = free.iter().map(|&i| relaxed.x_u[i]).collect();
        let d_lo: Vec<Number> = d_rows.iter().map(|&j| relaxed.g_l[j]).collect();
        let d_hi: Vec<Number> = d_rows.iter().map(|&j| relaxed.g_u[j]).collect();
        let xl_pos: Vec<usize> = (0..free.len())
            .filter(|&k| finite_lower(declared.x_l[free[k]]))
            .collect();
        let xu_pos: Vec<usize> = (0..free.len())
            .filter(|&k| finite_upper(declared.x_u[free[k]]))
            .collect();
        let dl_pos: Vec<usize> = (0..d_rows.len())
            .filter(|&k| finite_lower(declared.g_l[d_rows[k]]))
            .collect();
        let du_pos: Vec<usize> = (0..d_rows.len())
            .filter(|&k| finite_upper(declared.g_u[d_rows[k]]))
            .collect();

        let (n, m_d) = (free.len(), d_rows.len());
        let dims = [
            n,
            m_d,
            c_rows.len(),
            m_d,
            xl_pos.len(),
            xu_pos.len(),
            dl_pos.len(),
            du_pos.len(),
        ];
        let mut off = [0usize; 9];
        for i in 0..8 {
            off[i + 1] = off[i] + dims[i];
        }
        let mut bound_rows = Vec::new();
        for (k, &v) in xl_pos.iter().enumerate() {
            bound_rows.push(BoundRow {
                row: off[4] + k,
                var_row: v,
                lower: true,
            });
        }
        for (k, &v) in xu_pos.iter().enumerate() {
            bound_rows.push(BoundRow {
                row: off[5] + k,
                var_row: v,
                lower: false,
            });
        }
        for (k, &t) in dl_pos.iter().enumerate() {
            bound_rows.push(BoundRow {
                row: off[6] + k,
                var_row: n + t,
                lower: true,
            });
        }
        for (k, &t) in du_pos.iter().enumerate() {
            bound_rows.push(BoundRow {
                row: off[7] + k,
                var_row: n + t,
                lower: false,
            });
        }

        let mut eng = OutsideEngine {
            model,
            index_base,
            n_full,
            free,
            x_template: reported.x.clone(),
            c_rows,
            c_rhs,
            d_rows,
            full_g_to_d,
            crossover: false,
            x_decl_lo,
            x_decl_hi,
            d_decl_lo,
            d_decl_hi,
            x_lo,
            x_hi,
            d_lo,
            d_hi,
            xl_pos,
            xu_pos,
            dl_pos,
            du_pos,
            dims,
            off,
            bound_rows,
            jac_structure,
            hess_structure,
            base: Vec::new(),
            mu: reported.mu,
            slack_mu: reported.mu,
            obj_factor,
            conj: None,
            base_lu: Factored::new(0, Vec::new()).expect("empty factor"),
            fault: None,
        };
        if obj_factor != 1.0 {
            let o = eng.off;
            let (mut e, mut f) = (vec![1.0; o[8]], vec![1.0; o[8]]);
            for r in (o[0]..o[2]).chain(o[4]..o[8]) {
                e[r] = obj_factor;
            }
            for r in o[2]..o[8] {
                f[r] = 1.0 / obj_factor;
            }
            eng.conj = Some((e, f));
        }
        eng.mu = eng.barrier_level(reported);
        eng.base = eng.fill_in(reported);
        let k = eng.kkt_dense(&eng.base, &[], None);
        eng.base_lu =
            Factored::new(eng.dim(), k).expect("the KKT matrix at the converged point is singular");
        eng
    }

    /// The barrier level the reported point sits on.
    ///
    /// Normally the last iteration's `μ`: a converged barrier iterate holds
    /// every complementarity product near it. A terminating path that
    /// installs multipliers of its own can leave a point that is on no
    /// barrier path, upstream Ipopt's `ComputeFeasibilityMultipliers` on a
    /// square problem among them. It zeroes every bound multiplier, which
    /// no barrier iterate can do, since an interior-point engine holds
    /// `z > 0` strictly. pounce reads such a point at barrier level zero,
    /// and so does this engine, from what the report shows of it: every
    /// variable-bound multiplier exactly zero, and every one-sided row whose
    /// slack is positive holding a multiplier so small that slack times
    /// `|λ|` is under the `0.1·μ` the classification reads as off the
    /// central path on the low side. A row's multiplier comes to the report
    /// through stationarity rather than as itself, so it is roundoff rather
    /// than an exact zero there. A two-sided row is left out, since its `λ`
    /// is the difference of two multipliers and can be near zero on the
    /// path, and a slack of zero says nothing either way, which is where
    /// crossover parks a point with its multipliers intact.
    fn barrier_level(&self, r: &Reported) -> Number {
        let g = self.eval_g(&r.x);
        let mut evidence = 0usize;
        let mut on_path = false;
        for (v, side) in self
            .xl_pos
            .iter()
            .map(|&v| (v, &r.z_l))
            .chain(self.xu_pos.iter().map(|&v| (v, &r.z_u)))
        {
            evidence += 1;
            on_path |= side[self.free[v]] != 0.0;
        }
        for (t, &j) in self.d_rows.iter().enumerate() {
            let lo = self.dl_pos.contains(&t);
            let hi = self.du_pos.contains(&t);
            let slack = match (lo, hi) {
                (true, false) => g[j] - self.d_lo[t],
                (false, true) => self.d_hi[t] - g[j],
                _ => continue,
            };
            if slack > 0.0 {
                evidence += 1;
                on_path |= slack * r.lambda[j].abs() >= 0.1 * r.mu;
            }
        }
        if evidence > 0 && !on_path { 0.0 } else { r.mu }
    }

    /// The compound iterate a report implies.
    ///
    /// A report carries the primal point, one multiplier per constraint row
    /// and the variable-bound multipliers. The inequality slacks are the row
    /// values, which is where a converged slack sits. The row-bound
    /// multipliers satisfy `y_d = v_u − v_l`, the stationarity of the slack,
    /// where that keeps them non-negative, as a barrier engine holds them: a
    /// one-sided row's multiplier is `±y_d` floored at zero, and a row
    /// bounded on both sides puts the barrier's own `μ / s` on the side the
    /// sign of `y_d` says is inactive. `μ` here is the level
    /// [`Self::barrier_level`] decided, so a point on no barrier path gets
    /// zero there.
    fn fill_in(&self, r: &Reported) -> Vec<Number> {
        let o = self.off;
        let mut p = vec![0.0; self.dim()];
        for (k, &i) in self.free.iter().enumerate() {
            p[o[0] + k] = r.x[i];
        }
        let g = self.eval_g(&r.x);
        for (k, &j) in self.d_rows.iter().enumerate() {
            p[o[1] + k] = g[j];
        }
        for (k, &j) in self.c_rows.iter().enumerate() {
            p[o[2] + k] = r.lambda[j];
        }
        for (k, &j) in self.d_rows.iter().enumerate() {
            p[o[3] + k] = r.lambda[j];
        }
        for (k, &v) in self.xl_pos.iter().enumerate() {
            p[o[4] + k] = r.z_l[self.free[v]];
        }
        for (k, &v) in self.xu_pos.iter().enumerate() {
            p[o[5] + k] = r.z_u[self.free[v]];
        }
        for t in 0..self.d_rows.len() {
            let y = p[o[3] + t];
            let s = p[o[1] + t];
            let lo = self.dl_pos.iter().position(|&q| q == t);
            let hi = self.du_pos.iter().position(|&q| q == t);
            let (vl, vu) = match (lo, hi) {
                (Some(_), None) => ((-y).max(0.0), 0.0),
                (None, Some(_)) => (0.0, y.max(0.0)),
                (Some(_), Some(_)) => {
                    if y <= 0.0 {
                        let vu = self.mu / (self.d_hi[t] - s);
                        (vu - y, vu)
                    } else {
                        let vl = self.mu / (s - self.d_lo[t]);
                        (vl, vl + y)
                    }
                }
                (None, None) => (0.0, 0.0),
            };
            if let Some(k) = lo {
                p[o[6] + k] = vl;
            }
            if let Some(k) = hi {
                p[o[7] + k] = vu;
            }
        }
        p
    }

    /// Marks the report as a vertex crossover parked on its bounds.
    ///
    /// Such a point sits exactly on the bounds it is active at, so its
    /// barrier diagonal `z/s` divides by a slack that is zero. pounce reads
    /// that diagonal in the declared frame (gh#654), the slack to the bound
    /// the model declares floored at `ε·max(1, |b|)`, where its iterate's own
    /// slacks go through the safe-slack rule. An engine that ran crossover
    /// knows it did, and reads its diagonal the same way. Only the
    /// classification's diagonal changes: the slacks themselves, and
    /// everything the corrector reads, keep the safe-slack rule.
    pub fn after_crossover(mut self) -> Self {
        self.crossover = true;
        self
    }

    /// How far, relative to itself, each variable's and each row's barrier
    /// diagonal can move if the engine moved one of its bounds during the
    /// solve, by model index: the largest `slack_move·max(1, |b|)/s` over the
    /// entry's bounds, and zero where it has none.
    ///
    /// When a trial slack falls below `ε·min(1, μ)`, an Ipopt-family engine
    /// moves that bound outward by `slack_move·max(1, |b|)` and carries on
    /// (`AcceptTrialPoint`, which pounce ports). Which bounds moved depends
    /// on the path the solve took and no report carries it, so the engine's
    /// own slack and the one measured here differ by that move wherever it
    /// happened, which changes the bound's `z/s` by that much relative to
    /// itself. Far from the bound it is nothing. Within a few `slack_move` of
    /// it, as at a kink, it is most of the slack.
    pub fn bound_move_tolerance(&self) -> (Vec<Number>, Vec<Number>) {
        let slack_move = Number::EPSILON.powf(0.75);
        // A bound moved outward by `δ` leaves the engine's slack at `s + δ`
        // where this engine reads `s`, so the two diagonals `z/(s + δ)` and
        // `z/s` differ by `δ / (s + δ)` relative to the larger, below one.
        let rel = |b: Number, s: Number| {
            let delta = slack_move * b.abs().max(1.0);
            delta / (s + delta)
        };
        let p = &self.base;
        let o = self.off;
        let mut var = vec![0.0_f64; self.n_full];
        let mut row = vec![0.0_f64; self.c_rows.len() + self.d_rows.len()];
        for &v in &self.xl_pos {
            let t = rel(self.x_lo[v], p[o[0] + v] - self.x_lo[v]);
            var[self.free[v]] = var[self.free[v]].max(t);
        }
        for &v in &self.xu_pos {
            let t = rel(self.x_hi[v], self.x_hi[v] - p[o[0] + v]);
            var[self.free[v]] = var[self.free[v]].max(t);
        }
        for &t in &self.dl_pos {
            let r = rel(self.d_lo[t], p[o[1] + t] - self.d_lo[t]);
            row[self.d_rows[t]] = row[self.d_rows[t]].max(r);
        }
        for &t in &self.du_pos {
            let r = rel(self.d_hi[t], self.d_hi[t] - p[o[1] + t]);
            row[self.d_rows[t]] = row[self.d_rows[t]].max(r);
        }
        (var, row)
    }

    /// The relaxed box over the primal prefix `(x, s)`, the box the
    /// corrector clamps into.
    /// What a comparison of pounce's classification with this engine's
    /// allows for: [`Self::bound_move_tolerance`] and [`Self::pounce_ceiling`].
    pub fn allowance(&self) -> super::compare::Allowance {
        let (var_move, row_move) = self.bound_move_tolerance();
        let (var_cap, row_cap) = self.pounce_ceiling();
        super::compare::Allowance {
            var_move,
            row_move,
            var_cap,
            row_cap,
        }
    }

    /// The ceiling pounce puts on each barrier diagonal it factors (gh#737),
    /// per model variable and per model row, in the units a classification
    /// reports `Σ` in, `INFINITY` where none applies.
    ///
    /// pounce's cap on a variable is `(a/ε)(a/64)`, with `a` the largest
    /// magnitude in its column of the constraint Jacobian, and on a row's
    /// slack it is the same with `a = 1`, the slack's own coefficient. pounce
    /// takes `a` from its Jacobian in its own scaled frame, and this takes it
    /// from the model's Jacobian, so the cap computed here is pounce's only on
    /// a solve with no row or variable scaling. The report divides `Σ` by the
    /// objective factor, so the cap is divided by it too.
    pub fn pounce_ceiling(&self) -> (Vec<Number>, Vec<Number>) {
        let cap = ceiling;
        let fi = self.free_index();
        let mut a_max = vec![0.0_f64; self.free.len()];
        for (_, c, v) in self.eval_jac(&self.full_x(&self.base)) {
            if let Some(k) = fi[c] {
                a_max[k] = a_max[k].max(v.abs());
            }
        }
        let f = self.obj_factor.abs();
        let mut var = vec![Number::INFINITY; self.n_full];
        for (k, &i) in self.free.iter().enumerate() {
            var[i] = cap(a_max[k]) / f;
        }
        let mut row = vec![Number::INFINITY; self.c_rows.len() + self.d_rows.len()];
        for &j in &self.d_rows {
            row[j] = cap(1.0) / f;
        }
        (var, row)
    }

    /// The engine with `fault` in what it reports.
    pub fn with_fault(mut self, fault: Fault) -> Self {
        self.fault = Some(fault);
        self
    }

    pub fn primal_box(&self) -> (Vec<Number>, Vec<Number>) {
        let mut lo = self.x_lo.clone();
        lo.extend_from_slice(&self.d_lo);
        let mut hi = self.x_hi.clone();
        hi.extend_from_slice(&self.d_hi);
        (lo, hi)
    }

    /// The KKT row of the multiplier of each model equality row given, the
    /// rows a parametric step pins.
    pub fn pin_rows(&self, user_rows: &[Index]) -> Vec<usize> {
        user_rows
            .iter()
            .map(|&j| {
                let k = self
                    .c_rows
                    .iter()
                    .position(|&c| c == j as usize)
                    .expect("a pin is an equality row");
                self.off[2] + k
            })
            .collect()
    }

    pub fn converged(&self) -> &[Number] {
        &self.base
    }

    /// The converged primal point over the model's variables, in the
    /// model's order whatever the engine's layout.
    pub fn model_x(&self) -> Vec<Number> {
        self.full_x(&self.base)
    }

    pub fn mu(&self) -> Number {
        self.mu
    }

    // --- slacks -------------------------------------------------------------

    fn slack_xl(&self, p: &[Number], k: usize) -> Number {
        let v = self.xl_pos[k];
        let z = p[self.off[4] + k];
        safe_slack(
            p[self.off[0] + v] - self.x_lo[v],
            self.x_lo[v],
            z,
            self.slack_mu,
        )
    }

    fn slack_xu(&self, p: &[Number], k: usize) -> Number {
        let v = self.xu_pos[k];
        let z = p[self.off[5] + k];
        safe_slack(
            self.x_hi[v] - p[self.off[0] + v],
            self.x_hi[v],
            z,
            self.slack_mu,
        )
    }

    fn slack_dl(&self, p: &[Number], k: usize) -> Number {
        let t = self.dl_pos[k];
        let v = p[self.off[6] + k];
        safe_slack(
            p[self.off[1] + t] - self.d_lo[t],
            self.d_lo[t],
            v,
            self.slack_mu,
        )
    }

    fn slack_du(&self, p: &[Number], k: usize) -> Number {
        let t = self.du_pos[k];
        let v = p[self.off[7] + k];
        safe_slack(
            self.d_hi[t] - p[self.off[1] + t],
            self.d_hi[t],
            v,
            self.slack_mu,
        )
    }

    /// The barrier diagonal of a crossed-over point, in the declared frame:
    /// each bound's `z` over its slack to the declared bound, the slack
    /// floored at `ε·max(1, |b|)` and at `z_max/(f64::MAX/4)`, with `z_max`
    /// the largest multiplier in the block, so the sum stays representable.
    fn declared_sigma(&self, p: &[Number]) -> (Vec<Number>, Vec<Number>) {
        let o = self.off;
        let floor = |b: Number, z_max: Number| -> Number {
            let resolvable = Number::EPSILON * b.abs().max(1.0);
            if z_max.is_finite() && z_max > 0.0 {
                resolvable.max(z_max / (Number::MAX / 4.0))
            } else {
                resolvable
            }
        };
        let amax = |a: usize, b: usize| p[a..b].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let zx = amax(o[4], o[6]);
        let mut sx = vec![0.0; self.free.len()];
        for (k, &v) in self.xl_pos.iter().enumerate() {
            let s = (p[o[0] + v] - self.x_decl_lo[v]).max(floor(self.x_decl_lo[v], zx));
            sx[v] += p[o[4] + k] / s;
        }
        for (k, &v) in self.xu_pos.iter().enumerate() {
            let s = (self.x_decl_hi[v] - p[o[0] + v]).max(floor(self.x_decl_hi[v], zx));
            sx[v] += p[o[5] + k] / s;
        }
        let zs = amax(o[6], o[8]);
        let mut ss = vec![0.0; self.d_rows.len()];
        for (k, &t) in self.dl_pos.iter().enumerate() {
            let s = (p[o[1] + t] - self.d_decl_lo[t]).max(floor(self.d_decl_lo[t], zs));
            ss[t] += p[o[6] + k] / s;
        }
        for (k, &t) in self.du_pos.iter().enumerate() {
            let s = (self.d_decl_hi[t] - p[o[1] + t]).max(floor(self.d_decl_hi[t], zs));
            ss[t] += p[o[7] + k] / s;
        }
        (sx, ss)
    }

    // --- model evaluations ------------------------------------------------

    fn full_x(&self, p: &[Number]) -> Vec<Number> {
        let mut x = self.x_template.clone();
        for (k, &i) in self.free.iter().enumerate() {
            x[i] = p[self.off[0] + k];
        }
        x
    }

    fn full_lambda(&self, p: &[Number]) -> Vec<Number> {
        let mut l = vec![0.0; self.c_rows.len() + self.d_rows.len()];
        for (k, &j) in self.c_rows.iter().enumerate() {
            l[j] = p[self.off[2] + k];
        }
        for (k, &j) in self.d_rows.iter().enumerate() {
            l[j] = p[self.off[3] + k];
        }
        l
    }

    fn eval_g(&self, x: &[Number]) -> Vec<Number> {
        let m = self.c_rows.len() + self.d_rows.len();
        let mut g = vec![0.0; m];
        assert!(self.model.borrow_mut().eval_g(x, true, &mut g));
        g
    }

    fn eval_grad_f(&self, x: &[Number]) -> Vec<Number> {
        let mut g = vec![0.0; self.n_full];
        assert!(self.model.borrow_mut().eval_grad_f(x, true, &mut g));
        g
    }

    /// Jacobian entries `(row, model column, value)`, duplicates kept.
    fn eval_jac(&self, x: &[Number]) -> Vec<(usize, usize, Number)> {
        let mut v = vec![0.0; self.jac_structure.len()];
        assert!(self.model.borrow_mut().eval_jac_g(
            Some(x),
            true,
            SparsityRequest::Values { values: &mut v },
        ));
        self.jac_structure
            .iter()
            .zip(v)
            .map(|(&(r, c), val)| (r, c, val))
            .collect()
    }

    /// Hessian-of-the-Lagrangian entries `(row, column, value)` from one
    /// triangle, duplicates kept, with the engine's objective factor.
    fn eval_hess(&self, x: &[Number], lambda: &[Number]) -> Vec<(usize, usize, Number)> {
        if self.hess_structure.is_empty() {
            return Vec::new();
        }
        let mut v = vec![0.0; self.hess_structure.len()];
        assert!(self.model.borrow_mut().eval_h(
            Some(x),
            true,
            self.obj_factor,
            Some(lambda),
            true,
            SparsityRequest::Values { values: &mut v },
        ));
        self.hess_structure
            .iter()
            .zip(v)
            .map(|(&(r, c), val)| (r, c, val))
            .collect()
    }

    fn free_index(&self) -> Vec<Option<usize>> {
        let mut out = vec![None; self.n_full];
        for (k, &i) in self.free.iter().enumerate() {
            out[i] = Some(k);
        }
        out
    }

    // --- the KKT matrix ---------------------------------------------------

    /// The Newton matrix at compound point `p`, row-major, with each
    /// `(primal row, add)` in `pinned` added to its diagonal. `hess_at`
    /// overrides the point the Hessian is evaluated at.
    fn kkt_dense(
        &self,
        p: &[Number],
        pinned: &[(usize, Number)],
        hess_at: Option<&[Number]>,
    ) -> Vec<Number> {
        let dim = self.dim();
        let o = self.off;
        let mut k = vec![0.0; dim * dim];
        let mut put = |r: usize, c: usize, v: Number| k[r * dim + c] += v;
        let fi = self.free_index();

        let hp = hess_at.unwrap_or(p);
        let (xh, lh) = (self.full_x(hp), self.full_lambda(hp));
        for (i, j, v) in self.eval_hess(&xh, &lh) {
            if let (Some(a), Some(b)) = (fi[i], fi[j]) {
                put(o[0] + a, o[0] + b, v);
                if a != b {
                    put(o[0] + b, o[0] + a, v);
                }
            }
        }
        let x = self.full_x(p);
        for (r, c, v) in self.eval_jac(&x) {
            let Some(col) = fi[c] else { continue };
            if let Some(kc) = self.c_rows.iter().position(|&q| q == r) {
                put(o[2] + kc, o[0] + col, v);
                put(o[0] + col, o[2] + kc, v);
            } else if let Some(kd) = self.full_g_to_d[r] {
                put(o[3] + kd, o[0] + col, v);
                put(o[0] + col, o[3] + kd, v);
            }
        }
        for t in 0..self.d_rows.len() {
            put(o[1] + t, o[3] + t, -1.0);
            put(o[3] + t, o[1] + t, -1.0);
        }
        for (k_, &v) in self.xl_pos.iter().enumerate() {
            put(o[0] + v, o[4] + k_, -1.0);
            put(o[4] + k_, o[0] + v, p[o[4] + k_]);
            put(o[4] + k_, o[4] + k_, self.slack_xl(p, k_));
        }
        for (k_, &v) in self.xu_pos.iter().enumerate() {
            put(o[0] + v, o[5] + k_, 1.0);
            put(o[5] + k_, o[0] + v, -p[o[5] + k_]);
            put(o[5] + k_, o[5] + k_, self.slack_xu(p, k_));
        }
        for (k_, &t) in self.dl_pos.iter().enumerate() {
            put(o[1] + t, o[6] + k_, -1.0);
            put(o[6] + k_, o[1] + t, p[o[6] + k_]);
            put(o[6] + k_, o[6] + k_, self.slack_dl(p, k_));
        }
        for (k_, &t) in self.du_pos.iter().enumerate() {
            put(o[1] + t, o[7] + k_, 1.0);
            put(o[7] + k_, o[1] + t, -p[o[7] + k_]);
            put(o[7] + k_, o[7] + k_, self.slack_du(p, k_));
        }
        for &(row, add) in pinned {
            put(row, row, add);
        }
        k
    }

    /// The barrier residual at compound point `p`, unshifted.
    fn residual(&self, p: &[Number], out: &mut [Number]) {
        let o = self.off;
        let fi = self.free_index();
        let x = self.full_x(p);
        let gf = self.eval_grad_f(&x);
        let g = self.eval_g(&x);
        out.iter_mut().for_each(|v| *v = 0.0);
        for (k, &i) in self.free.iter().enumerate() {
            out[o[0] + k] = self.obj_factor * gf[i];
        }
        for (r, c, v) in self.eval_jac(&x) {
            let Some(col) = fi[c] else { continue };
            let y = if let Some(kc) = self.c_rows.iter().position(|&q| q == r) {
                p[o[2] + kc]
            } else if let Some(kd) = self.full_g_to_d[r] {
                p[o[3] + kd]
            } else {
                0.0
            };
            out[o[0] + col] += v * y;
        }
        for (k, &v) in self.xl_pos.iter().enumerate() {
            out[o[0] + v] -= p[o[4] + k];
        }
        for (k, &v) in self.xu_pos.iter().enumerate() {
            out[o[0] + v] += p[o[5] + k];
        }
        for t in 0..self.d_rows.len() {
            out[o[1] + t] = -p[o[3] + t];
        }
        for (k, &t) in self.dl_pos.iter().enumerate() {
            out[o[1] + t] -= p[o[6] + k];
        }
        for (k, &t) in self.du_pos.iter().enumerate() {
            out[o[1] + t] += p[o[7] + k];
        }
        for (k, &j) in self.c_rows.iter().enumerate() {
            out[o[2] + k] = g[j] - self.c_rhs[k];
        }
        for (k, &j) in self.d_rows.iter().enumerate() {
            out[o[3] + k] = g[j] - p[o[1] + k];
        }
        for k in 0..self.xl_pos.len() {
            out[o[4] + k] = self.slack_xl(p, k) * p[o[4] + k];
        }
        for k in 0..self.xu_pos.len() {
            out[o[5] + k] = self.slack_xu(p, k) * p[o[5] + k];
        }
        for k in 0..self.dl_pos.len() {
            out[o[6] + k] = self.slack_dl(p, k) * p[o[6] + k];
        }
        for k in 0..self.du_pos.len() {
            out[o[7] + k] = self.slack_du(p, k) * p[o[7] + k];
        }
    }
}

impl SensBacksolver for OutsideEngine {
    fn dim(&self) -> usize {
        self.off[8]
    }

    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        solve_through(&self.base_lu, self.conj.as_ref(), rhs, lhs)
    }

    fn natural_units_factor(&self) -> Option<&[Number]> {
        self.conj.as_ref().map(|(_, f)| f.as_slice())
    }

    fn bound_rows(&self) -> Option<&[BoundRow]> {
        Some(&self.bound_rows)
    }

    fn supports_release(&self) -> bool {
        true
    }

    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_with(released, &[], false, rhs, lhs)
    }

    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_with(released, &[], true, rhs, lhs)
    }

    fn solve_released_pinned(
        &self,
        released: &[usize],
        pinned: &[usize],
        rhs: &[Number],
        lhs: &mut [Number],
    ) -> bool {
        let n_p = self.dims[0] + self.dims[1];
        if pinned.is_empty() || pinned.iter().any(|&r| r >= n_p) {
            return false;
        }
        self.solve_released_with(released, pinned, false, rhs, lhs)
    }
}

impl OutsideEngine {
    /// A solve with the bounds `released` names taken out of the active set
    /// and each primal row in `pinned` held, as pounce's released solves
    /// answer.
    ///
    /// pounce factors the augmented system with each released bound's `z/s`
    /// left out of `Σ` and reads the released multiplier back from its own
    /// row. In the eight-block matrix that is the matrix with the released
    /// multiplier's entry in its primal row set to zero: the primal rows then
    /// carry no term of it, and its own row still gives it from the primal
    /// step. The right-hand side is zero on each released row, and `shift`
    /// moves the released multiplier onto its primal row, with the sign that
    /// row carries it with. A pin adds the ceiling pounce puts on that row to
    /// its diagonal.
    fn solve_released_with(
        &self,
        released: &[usize],
        pinned: &[usize],
        shift: bool,
        rhs: &[Number],
        lhs: &mut [Number],
    ) -> bool {
        let dim = self.dim();
        if rhs.len() != dim || lhs.len() != dim {
            return false;
        }
        if released.is_empty() && pinned.is_empty() {
            return self.solve(rhs, lhs);
        }
        let pins: Vec<(usize, Number)> = pinned.iter().map(|&r| (r, self.pin_add(r))).collect();
        let mut k = self.kkt_dense(&self.base, &pins, None);
        let mut r: Vec<Number> = match &self.conj {
            Some((e, _)) => rhs.iter().zip(e).map(|(&v, &ei)| v * ei).collect(),
            None => rhs.to_vec(),
        };
        for &row in released {
            let Some(b) = self.bound_rows.iter().find(|b| b.row == row) else {
                return false;
            };
            k[b.var_row * dim + row] = 0.0;
            r[row] = 0.0;
            if shift {
                let z = self.base[row];
                r[b.var_row] += if b.lower { -z } else { z };
            }
        }
        let Ok(lu) = Factored::new(dim, k) else {
            return false;
        };
        if !lu.solve(&r, lhs) {
            return false;
        }
        if let Some((_, f)) = &self.conj {
            for (l, &fi) in lhs.iter_mut().zip(f) {
                *l *= fi;
            }
        }
        true
    }

    /// The diagonal pounce adds to a pinned primal row: the ceiling of a
    /// variable's row, from the largest magnitude in its Jacobian column, and
    /// the ceiling with unit coefficient on a slack's row or on a variable in
    /// no constraint.
    fn pin_add(&self, row: usize) -> Number {
        if row < self.dims[0] {
            let col = self.free[row];
            let a = self
                .eval_jac(&self.full_x(&self.base))
                .into_iter()
                .filter(|&(_, c, _)| c == col)
                .fold(0.0_f64, |m, (_, _, v)| m.max(v.abs()));
            let cap = ceiling(a);
            if cap.is_finite() { cap } else { ceiling(1.0) }
        } else {
            ceiling(1.0)
        }
    }
}

/// pounce's ceiling on a barrier diagonal whose row's largest constraint
/// coefficient is `a` (gh#737): `(a/ε)(a/64)`, and `INFINITY` where none
/// applies.
fn ceiling(a: Number) -> Number {
    let c = (a / Number::EPSILON) * (a / 64.0);
    if a > 0.0 && a.is_finite() && c > 1.0 {
        c
    } else {
        Number::INFINITY
    }
}

/// A dense LU of the engine-frame `K̃`, with the matrix kept for iterative
/// refinement.
///
/// Ipopt and pounce both refine their solves against the matrix they
/// factored, and the engine does too. Without it the dense LU loses the
/// digits that a diagonal entry many orders above the rest of the matrix
/// costs, and the reduced curvature, a difference taken against such an
/// entry, keeps too few of them to compare with pounce's.
#[derive(Clone)]
struct Factored {
    n: usize,
    k: Vec<Number>,
    lu: DenseLuBacksolver,
}

impl Factored {
    /// The most refinement steps a solve takes.
    const MAX_REFINEMENTS: usize = 10;

    fn new(n: usize, k: Vec<Number>) -> Result<Self, ()> {
        let lu = DenseLuBacksolver::from_dense(n, &k)?;
        Ok(Self { n, k, lu })
    }

    fn residual(&self, rhs: &[Number], x: &[Number]) -> Vec<Number> {
        (0..self.n)
            .map(|i| {
                let row = &self.k[i * self.n..(i + 1) * self.n];
                rhs[i] - row.iter().zip(x).map(|(a, b)| a * b).sum::<Number>()
            })
            .collect()
    }

    /// Solves `K̃ x = rhs`, then refines `x` while each step at least halves
    /// the largest residual entry.
    fn solve(&self, rhs: &[Number], x: &mut [Number]) -> bool {
        if !self.lu.solve(rhs, x) {
            return false;
        }
        let norm = |v: &[Number]| v.iter().fold(0.0, |m: Number, a| m.max(a.abs()));
        let mut r = self.residual(rhs, x);
        let mut dx = vec![0.0; self.n];
        for _ in 0..Self::MAX_REFINEMENTS {
            let before = norm(&r);
            if before == 0.0 || !self.lu.solve(&r, &mut dx) {
                break;
            }
            let trial: Vec<Number> = x.iter().zip(&dx).map(|(a, d)| a + d).collect();
            let r_trial = self.residual(rhs, &trial);
            if norm(&r_trial) > 0.5 * before {
                if norm(&r_trial) < before {
                    x.copy_from_slice(&trial);
                }
                break;
            }
            x.copy_from_slice(&trial);
            r = r_trial;
        }
        true
    }
}

/// `K⁻¹·rhs` in natural units against a factor of the engine-frame `K̃`:
/// the right-hand side times `E`, the solve, and the result times `F`.
fn solve_through(
    lu: &Factored,
    conj: Option<&(Vec<Number>, Vec<Number>)>,
    rhs: &[Number],
    lhs: &mut [Number],
) -> bool {
    let r: Vec<Number> = match conj {
        Some((e, _)) => rhs.iter().zip(e).map(|(&v, &ei)| v * ei).collect(),
        None => rhs.to_vec(),
    };
    if !lu.solve(&r, lhs) {
        return false;
    }
    if let Some((_, f)) = conj {
        for (l, &fi) in lhs.iter_mut().zip(f) {
            *l *= fi;
        }
    }
    true
}

impl BarrierIterate for OutsideEngine {
    fn block_dims(&self) -> [usize; 8] {
        self.dims
    }

    fn barrier_mu(&self) -> Number {
        self.mu
    }

    fn scaled_rhs_factor(&self) -> Option<&[Number]> {
        self.conj.as_ref().map(|(e, _)| e.as_slice())
    }

    fn converged_iterate(&self, out: &mut [Number]) -> Result<(), String> {
        if out.len() != self.base.len() {
            return Err(format!(
                "outside engine: the iterate is {} long, not {}",
                self.base.len(),
                out.len()
            ));
        }
        out.copy_from_slice(&self.base);
        Ok(())
    }

    fn barrier_residual(&self, point: &[Number], out: &mut [Number]) -> Result<(), String> {
        if point.len() != self.dim() || out.len() != self.dim() {
            return Err("outside engine: residual called with the wrong length".into());
        }
        self.residual(point, out);
        Ok(())
    }

    fn corrector_operator<'a>(
        &'a self,
        predicted: &[Number],
        pinned: &[(usize, Number)],
        exact_hessian: bool,
    ) -> Result<Box<dyn KktOperator + 'a>, String> {
        let hess_at = if exact_hessian {
            None
        } else {
            Some(&self.base[..])
        };
        let at: Vec<Number> = match &self.conj {
            Some((_, f)) => predicted.iter().zip(f).map(|(&v, &fi)| v / fi).collect(),
            None => predicted.to_vec(),
        };
        // A pin's stiffness `μ/s²` is formed from the engine's own `μ` and
        // the primal slack, which the objective factor leaves alone, so it is
        // already an engine-frame diagonal value and goes in as it stands.
        let k = self.kkt_dense(&at, pinned, hess_at);
        let lu = Factored::new(self.dim(), k).map_err(|_| {
            "outside engine: the operator at the predicted point is singular".to_string()
        })?;
        Ok(Box::new(OutsideOperator {
            lu,
            conj: self.conj.clone(),
        }))
    }

    fn activity_data(&self) -> Result<ActivityData, String> {
        let o = self.off;
        let p = &self.base;
        let fi = self.free_index();
        let n = self.free.len();
        let m_d = self.d_rows.len();
        let side = |pos: &[usize], mult_at: usize, slack: &dyn Fn(usize) -> Number| BoundSide {
            pos: pos.to_vec(),
            mult: (0..pos.len()).map(|k| p[mult_at + k]).collect(),
            slack: (0..pos.len()).map(slack).collect(),
        };
        let x_l = side(&self.xl_pos, o[4], &|k| self.slack_xl(p, k));
        let x_u = side(&self.xu_pos, o[5], &|k| self.slack_xu(p, k));
        let d_l = side(&self.dl_pos, o[6], &|k| self.slack_dl(p, k));
        let d_u = side(&self.du_pos, o[7], &|k| self.slack_du(p, k));
        let (sigma_x, sigma_s) = if self.crossover {
            self.declared_sigma(p)
        } else {
            let mut sigma_x = vec![0.0; n];
            for s in [&x_l, &x_u] {
                for k in 0..s.pos.len() {
                    sigma_x[s.pos[k]] += s.mult[k] / s.slack[k];
                }
            }
            let mut sigma_s = vec![0.0; m_d];
            for s in [&d_l, &d_u] {
                for k in 0..s.pos.len() {
                    sigma_s[s.pos[k]] += s.mult[k] / s.slack[k];
                }
            }
            (sigma_x, sigma_s)
        };

        let (x, l) = (self.full_x(p), self.full_lambda(p));
        let (mut hr, mut hc, mut hv) = (Vec::new(), Vec::new(), Vec::new());
        for (i, j, v) in self.eval_hess(&x, &l) {
            if let (Some(a), Some(b)) = (fi[i], fi[j]) {
                hr.push(a.max(b) as Index + 1);
                hc.push(a.min(b) as Index + 1);
                hv.push(v);
            }
        }
        let mut hess = SymTMatrix::new(SymTMatrixSpace::new(n as Index, hr, hc));
        hess.set_values(&hv);
        let maps = self.model_maps()?;
        let jac_entries = self.eval_jac(&x);
        let block_jacobian = |map: &[Option<usize>], rows: usize| {
            let (mut jr, mut jc, mut jv) = (Vec::new(), Vec::new(), Vec::new());
            for &(r, c, v) in &jac_entries {
                if let (Some(t), Some(col)) = (map[r], fi[c]) {
                    jr.push(t as Index + 1);
                    jc.push(col as Index + 1);
                    jv.push(v);
                }
            }
            let mut jac = GenTMatrix::new(GenTMatrixSpace::new(rows as Index, n as Index, jr, jc));
            jac.set_values(&jv);
            jac
        };
        let jac_c = block_jacobian(&maps.full_g_to_c, self.c_rows.len());
        let jac = block_jacobian(&maps.full_g_to_d, m_d);

        Ok(ActivityData {
            x_l,
            x_u,
            d_l,
            d_u,
            sigma_x,
            sigma_s,
            hessian: Rc::new(hess) as Rc<dyn SymMatrix>,
            jac_c: Rc::new(jac_c) as Rc<dyn Matrix>,
            jac_d: Rc::new(jac) as Rc<dyn Matrix>,
        })
    }

    fn model_maps(&self) -> Result<ModelMaps, String> {
        let mut full_g_to_c = vec![None; self.c_rows.len() + self.d_rows.len()];
        for (k, &j) in self.c_rows.iter().enumerate() {
            full_g_to_c[j] = Some(k);
        }
        let mut var_x_to_full_x = self.free.clone();
        if self.fault == Some(Fault::Maps) && var_x_to_full_x.len() > 1 {
            var_x_to_full_x[1] = var_x_to_full_x[0];
        }
        Ok(ModelMaps {
            n_full_x: self.n_full,
            var_x_to_full_x,
            full_g_to_c,
            full_g_to_d: self.full_g_to_d.clone(),
        })
    }

    /// The objective factor is the engine's only scaling.
    fn scaling(&self) -> EngineScaling {
        let equality_rows = (self.fault == Some(Fault::Scaling))
            .then(|| vec![1.0; self.c_rows.len().saturating_sub(1)]);
        EngineScaling {
            objective: self.obj_factor,
            equality_rows,
            ..EngineScaling::none()
        }
    }

    fn primal_bounds(&self) -> Result<(Vec<Number>, Vec<Number>), String> {
        let (mut lo, mut hi) = self.primal_box();
        if self.fault == Some(Fault::PrimalBounds) {
            lo.pop();
            hi.pop();
        }
        Ok((lo, hi))
    }
}

/// The outside engine's operator at a predicted point: its own dense LU of
/// the Newton matrix there.
struct OutsideOperator {
    lu: Factored,
    conj: Option<(Vec<Number>, Vec<Number>)>,
}

impl KktOperator for OutsideOperator {
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        solve_through(&self.lu, self.conj.as_ref(), rhs, lhs)
    }
}

// ---------------------------------------------------------------------------
// Running the corrector and the classification over the outside engine
// ---------------------------------------------------------------------------

/// The bounds an engine iterated against, given the ones the model declares.
#[derive(Debug, Clone, Copy)]
pub enum Relaxation {
    /// `bound_relax_factor = 0`: the declared bounds.
    None,
    /// pounce's rule at `bound_relax_factor`, with `constr_viol_tol` as the cap.
    Pounce { factor: Number, cap: Number },
    /// Ipopt's rule at `bound_relax_factor`, with `constr_viol_tol` as the cap.
    Ipopt { factor: Number, cap: Number },
}

impl Relaxation {
    /// pounce's defaults: `bound_relax_factor = 1e-8`, `constr_viol_tol = 1e-4`.
    pub const POUNCE_DEFAULT: Relaxation = Relaxation::Pounce {
        factor: 1e-8,
        cap: 1e-4,
    };

    pub fn apply(self, declared: &Bounds) -> Bounds {
        match self {
            Relaxation::None => declared.clone(),
            Relaxation::Pounce { factor, cap } => declared.relaxed_as_pounce(factor, cap),
            Relaxation::Ipopt { factor, cap } => declared.relaxed_as_ipopt(factor, cap),
        }
    }
}

/// The outside engine over what a recorded solve returned.
pub fn outside_engine<T: TNLP + 'static>(
    model: &Rc<RefCell<RecordingTnlp<T>>>,
    relaxation: Relaxation,
) -> OutsideEngine {
    outside_engine_minimizing(model, relaxation, 1.0)
}

/// [`outside_engine`] with its blocks in `layout`.
pub fn outside_engine_with_layout<T: TNLP + 'static>(
    model: &Rc<RefCell<RecordingTnlp<T>>>,
    relaxation: Relaxation,
    layout: Layout,
) -> OutsideEngine {
    let reported = model
        .borrow()
        .reported
        .clone()
        .expect("the solve reported a solution");
    let declared = Bounds::declared(&mut *model.borrow_mut());
    let relaxed = relaxation.apply(&declared);
    let dyn_model: Rc<RefCell<dyn TNLP>> = model.clone();
    OutsideEngine::with_layout(dyn_model, &reported, &declared, &relaxed, 1.0, layout)
}

/// [`outside_engine`] for an engine that minimized `obj_factor·f`.
pub fn outside_engine_minimizing<T: TNLP + 'static>(
    model: &Rc<RefCell<RecordingTnlp<T>>>,
    relaxation: Relaxation,
    obj_factor: Number,
) -> OutsideEngine {
    let reported = model
        .borrow()
        .reported
        .clone()
        .expect("the solve reported a solution");
    let declared = Bounds::declared(&mut *model.borrow_mut());
    let relaxed = relaxation.apply(&declared);
    let dyn_model: Rc<RefCell<dyn TNLP>> = model.clone();
    OutsideEngine::with_objective_factor(dyn_model, &reported, &declared, &relaxed, obj_factor)
}

/// The corrector over the outside engine, with the perturbation given the
/// way `Solver::correct_step` takes it: model equality rows and the moves in
/// their right-hand sides.
pub fn correct_outside(
    eng: &OutsideEngine,
    pins: &[Index],
    deltas: &[Number],
    step: &[Number],
    max_iter: usize,
) -> Result<
    (Vec<Number>, pounce_sens_core::corrector::CorrectorReport),
    pounce_sens_core::corrector::CorrectorError,
> {
    correct_outside_holding(eng, pins, deltas, step, max_iter, true)
}

/// [`correct_outside`], with `exact_hessian = false` keeping the Hessian at
/// the converged point in the operator, as an engine that cannot evaluate one
/// elsewhere would.
pub fn correct_outside_holding(
    eng: &OutsideEngine,
    pins: &[Index],
    deltas: &[Number],
    step: &[Number],
    max_iter: usize,
    exact_hessian: bool,
) -> Result<
    (Vec<Number>, pounce_sens_core::corrector::CorrectorReport),
    pounce_sens_core::corrector::CorrectorError,
> {
    let (lo, hi) = eng.primal_box();
    let pin_rows = eng.pin_rows(pins);
    pounce_sens_core::corrector::correct(
        eng,
        step,
        &pin_rows,
        deltas,
        &lo,
        &hi,
        max_iter,
        exact_hessian,
    )
}

/// The parametric step over the outside engine, built the way
/// `Solver::parametric_step_full` builds pounce's: the sIPOPT step through
/// `SensApplication` against the engine's own factorization, plus the
/// barrier correction, a back-solve with `μ` in every complementarity row.
pub fn parametric_step_outside(
    eng: &OutsideEngine,
    pins: &[Index],
    deltas: &[Number],
) -> Vec<Number> {
    let rows: Vec<Index> = eng.pin_rows(pins).iter().map(|&r| r as Index).collect();
    let a = IndexSchurData::from_parts(rows, vec![1; pins.len()]).expect("pin rows");
    let opts = SensOptions {
        run_sens: true,
        ..SensOptions::default()
    };
    let app = SensApplication::new(a, eng.clone(), opts);
    let dim = eng.dim();
    let mut dx = vec![0.0; dim];
    assert!(app.parametric_step(deltas, &mut dx), "parametric step");
    let o = eng.offsets();
    let mut rhs = vec![0.0; dim];
    for r in &mut rhs[o[4]..o[8]] {
        *r = eng.mu();
    }
    let mut corr = vec![0.0; dim];
    assert!(eng.solve(&rhs, &mut corr), "barrier correction");
    for (d, c) in dx.iter_mut().zip(&corr) {
        *d += c * BARRIER_SIGN;
    }
    dx
}

/// Solves `model` under `app` with the model wrapped to record what the solve
/// returns, and hands back the solver, the wrapped model and the status.
pub fn recorded_solve<T: TNLP + 'static>(
    app: pounce_algorithm::application::IpoptApplication,
    model: T,
) -> (
    pounce_sensitivity::Solver,
    Rc<RefCell<RecordingTnlp<T>>>,
    pounce_nlp::return_codes::ApplicationReturnStatus,
) {
    let model = Rc::new(RefCell::new(RecordingTnlp::new(model)));
    let tnlp: Rc<RefCell<dyn TNLP>> = model.clone();
    let mut solver = pounce_sensitivity::Solver::new(app, tnlp);
    let status = solver.solve();
    (solver, model, status)
}

/// pounce's classification and the outside engine's over the same solve,
/// asserted to be the same.
pub fn same_classification<T: TNLP + 'static>(
    label: &str,
    solver: &pounce_sensitivity::Solver,
    model: &Rc<RefCell<RecordingTnlp<T>>>,
    relaxation: Relaxation,
) -> (
    pounce_sens_core::activity::ActivityReport,
    pounce_sens_core::activity::ActivityReport,
) {
    same_classification_of(label, solver, outside_engine(model, relaxation))
}

/// [`same_classification`] over an engine the caller built.
pub fn same_classification_of(
    label: &str,
    solver: &pounce_sensitivity::Solver,
    eng: OutsideEngine,
) -> (
    pounce_sens_core::activity::ActivityReport,
    pounce_sens_core::activity::ActivityReport,
) {
    let a = solver.classify_activity().expect("pounce's classification");
    let b = pounce_sens_core::activity::compute(&eng).expect("the outside classification");
    super::compare::assert_same_activity_within(label, &a, &b, &eng.allowance());
    (a, b)
}

/// pounce's reduced classification of `vars` and the outside engine's,
/// asserted to be the same.
pub fn same_reduced_activity(
    label: &str,
    solver: &pounce_sensitivity::Solver,
    eng: &OutsideEngine,
    vars: &[usize],
) {
    let a = solver
        .reduced_activity(vars)
        .expect("pounce's reduced classification");
    let b = pounce_sens_core::activity::reduced_activity(eng, vars)
        .expect("the outside reduced classification");
    super::compare::assert_same_reduced_activity(label, &a, &b);
}

/// pounce's reduced classification of `rows` and the outside engine's,
/// asserted to be the same.
pub fn same_reduced_row_activity(
    label: &str,
    solver: &pounce_sensitivity::Solver,
    eng: &OutsideEngine,
    rows: &[usize],
) {
    let a = solver
        .reduced_row_activity(rows)
        .expect("pounce's reduced row classification");
    let b = pounce_sens_core::activity::reduced_row_activity(eng, rows)
        .expect("the outside reduced row classification");
    super::compare::assert_same_reduced_row_activity(label, &a, &b);
}
