//! Adversary probe for Phase 6 — linear-equality variable elimination
//! (`presolve_linear_eq_reduction`, gh#487).
//!
//! Two independent attacks, because the pass has two independent ways to be
//! silently wrong.
//!
//! # 1. The plan (`plan` mode)
//!
//! Every instance is generated **around a known feasible point** `x*`: the
//! right-hand sides are computed from `x*`, and every box is drawn to contain
//! it. That makes `x*` an oracle the pass has never seen, and it pins six
//! properties that a wrong substitution cannot fake:
//!
//! * **Fidelity** — projecting `x*` onto the survivors and lifting must
//!   reproduce `x*` exactly. `x*` satisfies every row the pass consumed, so
//!   any substitution derived from those rows must reproduce it. A sign slip
//!   or a mis-composed chain shows up here immediately.
//! * **Feasible-point retention** — `x*`'s survivor coordinates must lie
//!   inside the *reduced* box. A bound transfer that is too aggressive would
//!   quietly exclude a known-feasible point, which is how a presolve turns a
//!   solvable model into a wrong "infeasible".
//! * **Box soundness** — for a *random* point of the reduced box, every
//!   lifted coordinate must sit inside its original declared box. This is the
//!   converse direction, and it is the one the `α < 0` bound flip breaks.
//! * **Row fidelity at an arbitrary point** — every consumed row must hold at
//!   the lift of a random reduced point, not just at `x*`.
//! * **Bound provenance** — every reduced bound must lift, through the
//!   recovery map of the column the plan says it came from, exactly onto that
//!   column's declared bound (gh#493). This is the identity postsolve relies
//!   on to hand a bound multiplier back to the column that owns it, and it is
//!   invisible to a stationarity check: full-space stationarity closes with
//!   the multiplier on either column.
//! * **Structural integrity** — rows and columns account exactly, and every
//!   `Affine` representative is a survivor (the wrapper's `lift_x` reads that
//!   slot directly).
//!
//! A consistent system inside a box containing `x*` must also never be called
//! infeasible.
//!
//! # 2. The derivative transforms (`deriv` mode)
//!
//! The wrapper turns `∇f`, `J`, and `∇²L` into precomputed scaled gathers.
//! The oracle is a **central finite difference of the wrapper's own reduced
//! `eval_f` / `eval_g`** — it knows nothing about the plan, and asks only the
//! question a wrong gather actually fails: is the published derivative the
//! derivative of the published function? Both the model and the substitution
//! are quadratic/affine, so the third derivative vanishes and the differences
//! are exact to roundoff.
//!
//! The structure matters as much as the values: an entry the gather never
//! emits reads as a structural zero forever. Densifying the published triplets
//! and comparing the *whole* matrix — not just the emitted positions — is what
//! catches that.
//!
//! An earlier version of this probe rebuilt the plan and compared against a
//! dense `AᵀHA`. It reported 483 failures out of 2000, every one of them the
//! probe's own fault: the pivot tie-break depends on the order the columns
//! arrive in, so the rebuilt plan was a different — equally valid — reduction
//! than the one the wrapper had chosen.

use pounce_common::types::{Index, Number};
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{
    LinearEqElimTnlp, PlanConfig, PlanInput, PresolveOptions, VarRecovery, build_plan,
};
use std::cell::RefCell;
use std::rc::Rc;

use crate::rng::Rng;

const SENTINEL_LO: Number = -1e19;
const SENTINEL_HI: Number = 1e19;

/// A randomly generated linear-equality system with a known feasible point.
pub struct Instance {
    pub seed: u64,
    pub n: usize,
    pub m: usize,
    pub rows: Vec<Vec<(usize, Number)>>,
    pub row_const: Vec<Number>,
    pub g_l: Vec<Number>,
    pub g_u: Vec<Number>,
    pub eligible: Vec<bool>,
    pub x_l: Vec<Number>,
    pub x_u: Vec<Number>,
    /// The point every row and every box was built around.
    pub x_star: Vec<Number>,
}

/// Generate a consistent instance around a known feasible point.
pub fn generate(r: &mut Rng, seed: u64) -> Instance {
    let n = r.int(3, 10);
    let m = r.int(1, 9);
    let x_star: Vec<Number> = (0..n).map(|_| r.range(-5.0, 5.0)).collect();

    let mut rows: Vec<Vec<(usize, Number)>> = Vec::with_capacity(m);
    let mut row_const = Vec::with_capacity(m);
    let mut g_l = Vec::with_capacity(m);
    let mut g_u = Vec::with_capacity(m);
    let mut eligible = Vec::with_capacity(m);

    for i in 0..m {
        // Bias hard toward the two-term shape — it is the one the pass exists
        // for — but keep singletons and three-term rows in the mix so the
        // "leave it alone" path is exercised too.
        let arity = if r.chance(0.15) {
            1
        } else if r.chance(0.75) {
            2
        } else {
            3
        }
        .min(n);

        // Duplicate an earlier row now and then, so substitutions render it
        // `0 = 0` and the redundant-row path fires.
        let entries = if i > 0 && r.chance(0.15) {
            let j = r.int(0, i - 1);
            rows[j].clone()
        } else {
            let mut cols: Vec<usize> = (0..n).collect();
            for k in (1..n).rev() {
                let j = r.int(0, k);
                cols.swap(k, j);
            }
            cols.truncate(arity);
            cols.iter()
                .map(|&c| {
                    // Keep coefficients away from zero: a coefficient the pass
                    // treats as structurally absent is a different test, and
                    // mixing the two would blur both.
                    let mag = r.range(0.3, 3.0);
                    (c, if r.chance(0.5) { mag } else { -mag })
                })
                .collect::<Vec<_>>()
        };

        let c = r.range(-2.0, 2.0);
        let body: Number = entries.iter().map(|&(j, a)| a * x_star[j]).sum::<Number>() + c;
        rows.push(entries);
        row_const.push(c);

        // Mostly equalities (consumable); sometimes an inequality or a row
        // tagged nonlinear, both of which the pass must leave alone.
        if r.chance(0.12) {
            g_l.push(body - r.range(0.5, 3.0));
            g_u.push(body + r.range(0.5, 3.0));
            eligible.push(true);
        } else {
            g_l.push(body);
            g_u.push(body);
            eligible.push(!r.chance(0.12));
        }
    }

    let mut x_l = vec![SENTINEL_LO; n];
    let mut x_u = vec![SENTINEL_HI; n];
    for j in 0..n {
        if r.chance(0.08) {
            // Fixed by its own box — shape 1.
            x_l[j] = x_star[j];
            x_u[j] = x_star[j];
        } else {
            if r.chance(0.7) {
                x_l[j] = x_star[j] - r.range(0.0, 4.0);
            }
            if r.chance(0.7) {
                x_u[j] = x_star[j] + r.range(0.0, 4.0);
            }
        }
    }

    Instance {
        seed,
        n,
        m,
        rows,
        row_const,
        g_l,
        g_u,
        eligible,
        x_l,
        x_u,
        x_star,
    }
}

impl Instance {
    fn plan_input(&self) -> PlanInput<'_> {
        PlanInput {
            n_vars: self.n,
            n_rows: self.m,
            rows: &self.rows,
            row_const: &self.row_const,
            g_l: &self.g_l,
            g_u: &self.g_u,
            eligible: &self.eligible,
            x_l: &self.x_l,
            x_u: &self.x_u,
        }
    }

    fn row_body(&self, i: usize, x: &[Number]) -> Number {
        self.rows[i].iter().map(|&(j, a)| a * x[j]).sum::<Number>() + self.row_const[i]
    }

    fn scale(&self) -> Number {
        let a = self
            .rows
            .iter()
            .flat_map(|r| r.iter().map(|&(_, v)| v.abs()))
            .fold(1.0_f64, f64::max);
        let x = self.x_star.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        a * x
    }
}

/// What a single plan-mode instance concluded.
pub enum PlanVerdict {
    Ok { elim_vars: usize, elim_rows: usize },
    Failed(String),
}

pub fn check_plan(inst: &Instance) -> PlanVerdict {
    let plan = build_plan(&inst.plan_input(), &PlanConfig::default());
    let tol = 1e-7 * inst.scale();

    // A consistent system, inside a box that demonstrably contains a feasible
    // point, must never be called contradictory.
    if plan.report.infeasible {
        return PlanVerdict::Failed(format!(
            "declared the equality system contradictory, but x* = {:?} satisfies every row \
             inside its own box",
            inst.x_star
        ));
    }

    // --- structural integrity -------------------------------------------
    let n_kept_rows = plan.rows_kept.len();
    let accounted = n_kept_rows + plan.report.n_rows_eliminated + plan.report.n_redundant_rows;
    if accounted != inst.m {
        return PlanVerdict::Failed(format!(
            "row accounting: {n_kept_rows} kept + {} consumed + {} redundant != {} rows",
            plan.report.n_rows_eliminated, plan.report.n_redundant_rows, inst.m
        ));
    }
    let n_elim_vars = plan
        .recovery
        .iter()
        .filter(|r| !matches!(r, VarRecovery::Kept(_)))
        .count();
    if plan.vars_kept.len() + n_elim_vars != inst.n {
        return PlanVerdict::Failed(format!(
            "column accounting: {} kept + {n_elim_vars} eliminated != {} columns",
            plan.vars_kept.len(),
            inst.n
        ));
    }
    for (i, rec) in plan.recovery.iter().enumerate() {
        match *rec {
            VarRecovery::Affine { rep, coeff, .. } => {
                if !matches!(plan.recovery[rep], VarRecovery::Kept(_)) {
                    return PlanVerdict::Failed(format!(
                        "column {i} is affine on column {rep}, which is not a survivor"
                    ));
                }
                if coeff == 0.0 || !coeff.is_finite() {
                    return PlanVerdict::Failed(format!("column {i} has coefficient {coeff}"));
                }
            }
            VarRecovery::Constant(c) if !c.is_finite() => {
                return PlanVerdict::Failed(format!("column {i} pinned to {c}"));
            }
            _ => {}
        }
    }
    if plan.vars_kept.len() != plan.x_l_red.len() || plan.vars_kept.len() != plan.x_u_red.len() {
        return PlanVerdict::Failed("reduced box length disagrees with the survivor count".into());
    }

    // --- fidelity: lifting x*'s survivors must reproduce x* --------------
    let mut y_star = vec![0.0; plan.vars_kept.len()];
    plan.project_x(&inst.x_star, &mut y_star);
    let mut x_hat = vec![0.0; inst.n];
    plan.lift_x(&y_star, &mut x_hat);
    for j in 0..inst.n {
        if (x_hat[j] - inst.x_star[j]).abs() > tol {
            return PlanVerdict::Failed(format!(
                "lift(project(x*)) changed column {j}: {} -> {} (x* = {:?})",
                inst.x_star[j], x_hat[j], inst.x_star
            ));
        }
    }

    // --- the reduced box must still admit the known feasible point -------
    for (red, &full) in plan.vars_kept.iter().enumerate() {
        let (lo, hi) = (plan.x_l_red[red], plan.x_u_red[red]);
        let v = inst.x_star[full];
        if lo > SENTINEL_LO && v < lo - tol {
            return PlanVerdict::Failed(format!(
                "reduced lower bound on column {full} is {lo}, above the feasible x*[{full}] = {v}"
            ));
        }
        if hi < SENTINEL_HI && v > hi + tol {
            return PlanVerdict::Failed(format!(
                "reduced upper bound on column {full} is {hi}, below the feasible x*[{full}] = {v}"
            ));
        }
    }

    // --- bound provenance (gh#493) ---------------------------------------
    // Postsolve hands a reduced bound's multiplier to the column the bound
    // was declared on, rescaled by that column's coefficient and flipped to
    // the other side of its box when the coefficient is negative. All of
    // that rests on one identity, which is what this checks: lifting the
    // reduced bound through the origin's own recovery map must land exactly
    // on the origin's declared bound. The stationarity checks elsewhere in
    // this probe cannot see a wrong answer here — they pass with the
    // multiplier on either column.
    if plan.vars_kept.len() != plan.x_l_src.len() || plan.vars_kept.len() != plan.x_u_src.len() {
        return PlanVerdict::Failed(
            "bound provenance length disagrees with the survivor count".into(),
        );
    }
    for (red, &full) in plan.vars_kept.iter().enumerate() {
        for (src, bound, upper) in [
            (plan.x_l_src[red], plan.x_l_red[red], false),
            (plan.x_u_src[red], plan.x_u_red[red], true),
        ] {
            if src >= inst.n {
                return PlanVerdict::Failed(format!(
                    "bound provenance on column {full} names column {src}, out of range"
                ));
            }
            // An absent bound carries no multiplier, so it needs no owner.
            if bound <= SENTINEL_LO || bound >= SENTINEL_HI || !bound.is_finite() {
                continue;
            }
            let (coeff, offset) = if src == full {
                (1.0, 0.0)
            } else {
                match plan.recovery[src] {
                    VarRecovery::Affine { rep, coeff, offset } if rep == full => (coeff, offset),
                    other => {
                        return PlanVerdict::Failed(format!(
                            "column {full}'s {} bound is attributed to column {src}, whose \
                             recovery is {other:?} rather than an affine image of {full}",
                            if upper { "upper" } else { "lower" }
                        ));
                    }
                }
            };
            // Which side of the origin's box: its own when α > 0, the
            // opposite one when α < 0.
            let declared = if upper != (coeff < 0.0) {
                inst.x_u[src]
            } else {
                inst.x_l[src]
            };
            if declared <= SENTINEL_LO || declared >= SENTINEL_HI {
                return PlanVerdict::Failed(format!(
                    "column {full}'s finite {} bound {bound} is attributed to column {src}, \
                     which declares no bound on that side",
                    if upper { "upper" } else { "lower" }
                ));
            }
            let lifted = coeff * bound + offset;
            if (lifted - declared).abs() > tol {
                return PlanVerdict::Failed(format!(
                    "column {full}'s {} bound {bound} lifts to {lifted} on column {src}, whose \
                     declared bound there is {declared}",
                    if upper { "upper" } else { "lower" }
                ));
            }
        }
    }

    // --- soundness at an arbitrary point of the reduced box --------------
    // Sample the corners and the interior; the corners are where a bound
    // transfer with the wrong sign shows itself.
    let mut probe = Rng::new(inst.seed ^ 0x5eed_1e11);
    for trial in 0..8 {
        let mut y = vec![0.0; plan.vars_kept.len()];
        for (red, slot) in y.iter_mut().enumerate() {
            let lo = if plan.x_l_red[red] > SENTINEL_LO {
                plan.x_l_red[red]
            } else {
                -50.0
            };
            let hi = if plan.x_u_red[red] < SENTINEL_HI {
                plan.x_u_red[red]
            } else {
                50.0
            };
            let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
            *slot = match trial % 4 {
                0 => lo,
                1 => hi,
                _ => probe.range(lo, hi),
            };
        }
        let mut x = vec![0.0; inst.n];
        plan.lift_x(&y, &mut x);

        // Every lifted coordinate inside its ORIGINAL declared box.
        for j in 0..inst.n {
            let slack = 1e-7 * x[j].abs().max(1.0);
            if inst.x_l[j] > SENTINEL_LO && x[j] < inst.x_l[j] - slack {
                return PlanVerdict::Failed(format!(
                    "a point of the reduced box lifts to x[{j}] = {} below its declared lower \
                     bound {} (trial {trial})",
                    x[j], inst.x_l[j]
                ));
            }
            if inst.x_u[j] < SENTINEL_HI && x[j] > inst.x_u[j] + slack {
                return PlanVerdict::Failed(format!(
                    "a point of the reduced box lifts to x[{j}] = {} above its declared upper \
                     bound {} (trial {trial})",
                    x[j], inst.x_u[j]
                ));
            }
        }

        // Every consumed row satisfied at the lifted point.
        let live = x.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for i in 0..inst.m {
            if plan.row_kept[i] {
                continue;
            }
            let body = inst.row_body(i, &x);
            let resid = (body - inst.g_l[i]).abs();
            if resid > 1e-7 * live * inst.scale().max(1.0) {
                return PlanVerdict::Failed(format!(
                    "consumed row {i} violated by {resid:.3e} at a point of the reduced box \
                     (trial {trial})"
                ));
            }
        }
    }

    PlanVerdict::Ok {
        elim_vars: n_elim_vars,
        elim_rows: inst.m - n_kept_rows,
    }
}

// ---------------------------------------------------------------------------
// Derivative-transform probe
// ---------------------------------------------------------------------------

/// `min ½ xᵀQx + cᵀx` subject to the instance's linear rows plus one dense
/// quadratic row, published with a dense Jacobian and a dense lower-triangular
/// Hessian so the wrapper's mapping has nowhere to hide.
struct QuadTnlp {
    n: usize,
    m: usize,
    q: Vec<Number>,
    c: Vec<Number>,
    /// Symmetric matrix of each nonlinear row (all-zero for the linear ones).
    r_quad: Vec<Vec<Number>>,
    lin: Vec<Vec<(usize, Number)>>,
    lin_const: Vec<Number>,
    is_lin: Vec<bool>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    x0: Vec<Number>,
    /// Whatever the wrapper handed down at the last `finalize_solution`.
    captured_x: Option<Vec<Number>>,
}

impl QuadTnlp {
    fn grad_f(&self, x: &[Number], out: &mut [Number]) {
        for i in 0..self.n {
            let mut s = self.c[i];
            for j in 0..self.n {
                s += self.q[i * self.n + j] * x[j];
            }
            out[i] = s;
        }
    }
    fn jac_dense(&self, x: &[Number]) -> Vec<Number> {
        let mut j = vec![0.0; self.m * self.n];
        for r in 0..self.m {
            if self.is_lin[r] {
                for &(col, a) in &self.lin[r] {
                    j[r * self.n + col] += a;
                }
            } else {
                for i in 0..self.n {
                    let mut s = 0.0;
                    for k in 0..self.n {
                        s += self.r_quad[r][i * self.n + k] * x[k];
                    }
                    j[r * self.n + i] = s;
                }
            }
        }
        j
    }
    /// Dense symmetric `∇²L = obj·Q + Σ λ_r R_r`.
    fn hess_dense(&self, obj: Number, lambda: &[Number]) -> Vec<Number> {
        let mut h = vec![0.0; self.n * self.n];
        for k in 0..self.n * self.n {
            h[k] = obj * self.q[k];
        }
        for r in 0..self.m {
            if self.is_lin[r] {
                continue;
            }
            for k in 0..self.n * self.n {
                h[k] += lambda[r] * self.r_quad[r][k];
            }
        }
        h
    }
}

impl TNLP for QuadTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as Index,
            m: self.m as Index,
            nnz_jac_g: (self.m * self.n) as Index,
            nnz_h_lag: (self.n * (self.n + 1) / 2) as Index,
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
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        for (r, t) in types.iter_mut().enumerate() {
            *t = if self.is_lin[r] {
                Linearity::Linear
            } else {
                Linearity::NonLinear
            };
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut s = 0.0;
        for i in 0..self.n {
            s += self.c[i] * x[i];
            for j in 0..self.n {
                s += 0.5 * self.q[i * self.n + j] * x[i] * x[j];
            }
        }
        Some(s)
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        self.grad_f(x, g);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for r in 0..self.m {
            g[r] = if self.is_lin[r] {
                self.lin[r].iter().map(|&(j, a)| a * x[j]).sum::<Number>() + self.lin_const[r]
            } else {
                let mut s = 0.0;
                for i in 0..self.n {
                    for k in 0..self.n {
                        s += 0.5 * self.r_quad[r][i * self.n + k] * x[i] * x[k];
                    }
                }
                s
            };
        }
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for r in 0..self.m {
                    for c in 0..self.n {
                        irow[k] = r as Index;
                        jcol[k] = c as Index;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                values.copy_from_slice(&self.jac_dense(x));
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for r in 0..self.n {
                    for c in 0..=r {
                        irow[k] = r as Index;
                        jcol[k] = c as Index;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let zero = vec![0.0; self.m];
                let lam = lambda.unwrap_or(&zero);
                let h = self.hess_dense(obj_factor, lam);
                let mut k = 0;
                for r in 0..self.n {
                    for c in 0..=r {
                        values[k] = h[r * self.n + c];
                        k += 1;
                    }
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.captured_x = Some(s.x.to_vec());
    }
}

fn sym_matrix(r: &mut Rng, n: usize) -> Vec<Number> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = r.range(-2.0, 2.0);
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    m
}

pub fn check_derivatives(inst: &Instance, r: &mut Rng) -> Result<usize, String> {
    let n = inst.n;
    // Reuse the instance's linear rows, and append one dense quadratic row so
    // the Hessian has a constraint contribution to carry through.
    let mut lin = inst.rows.clone();
    let mut lin_const = inst.row_const.clone();
    let mut is_lin: Vec<bool> = inst.eligible.clone();
    let mut g_l = inst.g_l.clone();
    let mut g_u = inst.g_u.clone();
    let mut r_quad: Vec<Vec<Number>> = (0..inst.m).map(|_| vec![0.0; n * n]).collect();

    lin.push(Vec::new());
    lin_const.push(0.0);
    is_lin.push(false);
    r_quad.push(sym_matrix(r, n));
    g_l.push(-1e19);
    g_u.push(1e19);
    let m = inst.m + 1;

    for i in 0..inst.m {
        if !inst.eligible[i] {
            r_quad[i] = vec![0.0; n * n];
        }
    }

    let q = sym_matrix(r, n);
    let c: Vec<Number> = (0..n).map(|_| r.range(-2.0, 2.0)).collect();

    let tnlp = Rc::new(RefCell::new(QuadTnlp {
        n,
        m,
        q,
        c,
        r_quad,
        lin,
        lin_const,
        is_lin,
        g_l,
        g_u,
        x_l: inst.x_l.clone(),
        x_u: inst.x_u.clone(),
        x0: inst.x_star.clone(),
        captured_x: None,
    }));

    let opts = PresolveOptions {
        enabled: true,
        linear_eq_reduction: true,
        ..PresolveOptions::defaults()
    };
    let mut w = LinearEqElimTnlp::new(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>, opts);
    let Some(info) = w.get_nlp_info() else {
        return Err("the wrapper declined to publish dimensions".into());
    };
    let n_red = info.n as usize;
    let m_red = info.m as usize;
    if n_red == n {
        return Ok(0); // nothing eliminated: the passthrough path, not this probe's target
    }

    // The oracle is a central finite difference of the wrapper's OWN reduced
    // `eval_f` / `eval_g`. That deliberately knows nothing about the plan: it
    // asks only "is the published derivative the derivative of the published
    // function?", which is the question a wrong gather actually fails. (An
    // earlier version of this probe rebuilt the plan and compared against a
    // dense `AᵀHA`; it reported hundreds of false positives, because the
    // pivot tie-break depends on column order and the rebuilt plan was a
    // different — equally valid — reduction.)
    //
    // Both the model and the substitution are quadratic/affine, so the third
    // derivative vanishes and central differences are exact to roundoff.
    let y: Vec<Number> = (0..n_red).map(|_| r.range(-3.0, 3.0)).collect();
    let lambda_red: Vec<Number> = (0..m_red).map(|_| r.range(-1.5, 1.5)).collect();
    let obj_factor = r.range(0.25, 2.0);
    let h = 1e-4;

    let jac_nnz = info.nnz_jac_g as usize;
    let mut irow = vec![0 as Index; jac_nnz];
    let mut jcol = vec![0 as Index; jac_nnz];
    if !w.eval_jac_g(
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut irow,
            jcol: &mut jcol,
        },
    ) {
        return Err("eval_jac_g(Structure) declined".into());
    }

    // Reduced gradient of the Lagrangian at an arbitrary reduced point, built
    // only from published quantities.
    let mut lag_grad = |w: &mut LinearEqElimTnlp, yy: &[Number]| -> Result<Vec<Number>, String> {
        let mut g = vec![0.0; n_red];
        if !w.eval_grad_f(yy, true, &mut g) {
            return Err("eval_grad_f declined".into());
        }
        for v in g.iter_mut() {
            *v *= obj_factor;
        }
        let mut vals = vec![0.0; jac_nnz];
        if !w.eval_jac_g(Some(yy), true, SparsityRequest::Values { values: &mut vals }) {
            return Err("eval_jac_g(Values) declined".into());
        }
        for k in 0..jac_nnz {
            let (rr, cc) = (irow[k] as usize, jcol[k] as usize);
            g[cc] += lambda_red[rr] * vals[k];
        }
        Ok(g)
    };

    let scale_of = |v: &[Number]| v.iter().fold(1.0_f64, |m, x| m.max(x.abs()));

    // --- gradient vs central difference of eval_f -----------------------
    let mut got_grad = vec![0.0; n_red];
    if !w.eval_grad_f(&y, true, &mut got_grad) {
        return Err("eval_grad_f declined".into());
    }
    for j in 0..n_red {
        let mut yp = y.clone();
        let mut ym = y.clone();
        yp[j] += h;
        ym[j] -= h;
        let fp = w.eval_f(&yp, true).ok_or("eval_f declined")?;
        let fm = w.eval_f(&ym, true).ok_or("eval_f declined")?;
        let fd = (fp - fm) / (2.0 * h);
        if (got_grad[j] - fd).abs() > 1e-5 * fd.abs().max(scale_of(&got_grad)) {
            return Err(format!(
                "reduced gradient[{j}] = {} but a central difference of the wrapper's own \
                 eval_f gives {fd}",
                got_grad[j]
            ));
        }
    }

    // --- Jacobian vs central difference of eval_g -----------------------
    let mut vals = vec![0.0; jac_nnz];
    if !w.eval_jac_g(Some(&y), true, SparsityRequest::Values { values: &mut vals }) {
        return Err("eval_jac_g(Values) declined".into());
    }
    let mut got_jac = vec![0.0; m_red * n_red];
    for k in 0..jac_nnz {
        let (rr, cc) = (irow[k] as usize, jcol[k] as usize);
        if rr >= m_red || cc >= n_red {
            return Err(format!("Jacobian entry {k} at ({rr},{cc}) is out of range"));
        }
        got_jac[rr * n_red + cc] += vals[k];
    }
    let mut fd_jac = vec![0.0; m_red * n_red];
    for j in 0..n_red {
        let mut yp = y.clone();
        let mut ym = y.clone();
        yp[j] += h;
        ym[j] -= h;
        let (mut gp, mut gm) = (vec![0.0; m_red], vec![0.0; m_red]);
        if !w.eval_g(&yp, true, &mut gp) || !w.eval_g(&ym, true, &mut gm) {
            return Err("eval_g declined".into());
        }
        for rr in 0..m_red {
            fd_jac[rr * n_red + j] = (gp[rr] - gm[rr]) / (2.0 * h);
        }
    }
    let jscale = scale_of(&fd_jac);
    for rr in 0..m_red {
        for cc in 0..n_red {
            let (a, b) = (got_jac[rr * n_red + cc], fd_jac[rr * n_red + cc]);
            if (a - b).abs() > 1e-5 * jscale {
                return Err(format!(
                    "reduced Jacobian[{rr}][{cc}] = {a} but a central difference of the \
                     wrapper's own eval_g gives {b}"
                ));
            }
        }
    }

    // --- Hessian vs central difference of the reduced Lagrangian gradient
    let h_nnz = info.nnz_h_lag as usize;
    let mut hirow = vec![0 as Index; h_nnz];
    let mut hjcol = vec![0 as Index; h_nnz];
    if !w.eval_h(
        None,
        false,
        1.0,
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut hirow,
            jcol: &mut hjcol,
        },
    ) {
        return Err("eval_h(Structure) declined".into());
    }
    let mut hvals = vec![0.0; h_nnz];
    if !w.eval_h(
        Some(&y),
        true,
        obj_factor,
        Some(&lambda_red),
        true,
        SparsityRequest::Values {
            values: &mut hvals,
        },
    ) {
        return Err("eval_h(Values) declined".into());
    }
    // Densify the published lower triangle into a full symmetric matrix.
    let mut got_h = vec![0.0; n_red * n_red];
    for k in 0..h_nnz {
        let (rr, cc) = (hirow[k] as usize, hjcol[k] as usize);
        if rr >= n_red || cc >= n_red {
            return Err(format!("Hessian entry {k} at ({rr},{cc}) is out of range"));
        }
        if cc > rr {
            return Err(format!(
                "Hessian entry {k} at ({rr},{cc}) is above the diagonal; the contract is the \
                 lower triangle"
            ));
        }
        got_h[rr * n_red + cc] += hvals[k];
        if rr != cc {
            got_h[cc * n_red + rr] += hvals[k];
        }
    }
    let mut fd_h = vec![0.0; n_red * n_red];
    for j in 0..n_red {
        let mut yp = y.clone();
        let mut ym = y.clone();
        yp[j] += h;
        ym[j] -= h;
        let gp = lag_grad(&mut w, &yp)?;
        let gm = lag_grad(&mut w, &ym)?;
        for i in 0..n_red {
            fd_h[i * n_red + j] = (gp[i] - gm[i]) / (2.0 * h);
        }
    }
    let hscale = scale_of(&fd_h);
    for rr in 0..n_red {
        for cc in 0..n_red {
            let (a, b) = (got_h[rr * n_red + cc], fd_h[rr * n_red + cc]);
            if (a - b).abs() > 1e-5 * hscale {
                return Err(format!(
                    "reduced Hessian[{rr}][{cc}] = {a} but a central difference of the reduced \
                     Lagrangian gradient gives {b}"
                ));
            }
        }
    }

    // --- primal fidelity: the consumed rows must hold at the lift of an
    // ARBITRARY reduced point ---------------------------------------------
    //
    // Derivatives cannot see this. The substitution is affine, and an error in
    // its constant term `β` shifts the lifted point without touching a single
    // derivative — `∇(αy + β)` is `α` either way. So a wrong row constant
    // (`g_r(x) = Σ a_j x_j + c`, read off the probe point) produces a reduced
    // problem whose solutions violate the very rows it eliminated, and every
    // finite-difference check above passes anyway. Ask the question directly:
    // push a reduced point through `finalize_solution` and look at what the
    // original model receives.
    //
    // At least as many original rows must hold at BOTH of two different lifts
    // as the reduction claims to have dropped. A kept row that happens to hold
    // only inflates the count, so the bound is safe in the direction it is
    // asserted.
    let mut identically_satisfied = 0usize;
    let mut resid_ok: Vec<bool> = vec![true; m];
    for trial in 0..2 {
        let yy: Vec<Number> = (0..n_red).map(|_| r.range(-4.0, 4.0)).collect();
        let zeros_n = vec![0.0; n_red];
        let zeros_m = vec![0.0; m_red];
        w.finalize_solution(
            Solution {
                status: SolverReturn::Success,
                x: &yy,
                z_l: &zeros_n,
                z_u: &zeros_n,
                g: &zeros_m,
                lambda: &zeros_m,
                obj_value: 0.0,
            },
            &IpoptData::default(),
            &IpoptCq::default(),
        );
        let x_full = tnlp
            .borrow()
            .captured_x
            .clone()
            .ok_or("the wrapper never forwarded finalize_solution")?;
        if x_full.len() != n {
            return Err(format!(
                "finalize_solution forwarded {} coordinates for an {n}-column model",
                x_full.len()
            ));
        }
        let live = x_full.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for i in 0..m {
            let is_eq = is_lin_of(&tnlp, i) && (g_of(&tnlp, i).1 - g_of(&tnlp, i).0).abs() <= 1e-12;
            if !is_eq {
                resid_ok[i] = false;
                continue;
            }
            let body: Number = row_body_of(&tnlp, i, &x_full);
            if (body - g_of(&tnlp, i).0).abs() > 1e-7 * live * 10.0 {
                resid_ok[i] = false;
            }
        }
        let _ = trial;
    }
    for i in 0..m {
        if resid_ok[i] {
            identically_satisfied += 1;
        }
    }
    let dropped = m - m_red;
    if identically_satisfied < dropped {
        return Err(format!(
            "the reduction dropped {dropped} rows, but only {identically_satisfied} of the \
             original rows hold at the lift of an arbitrary reduced point — the eliminated \
             rows are not satisfied by the substitution"
        ));
    }

    Ok(n - n_red)
}

fn is_lin_of(t: &Rc<RefCell<QuadTnlp>>, i: usize) -> bool {
    t.borrow().is_lin[i]
}

fn g_of(t: &Rc<RefCell<QuadTnlp>>, i: usize) -> (Number, Number) {
    let b = t.borrow();
    (b.g_l[i], b.g_u[i])
}

fn row_body_of(t: &Rc<RefCell<QuadTnlp>>, i: usize, x: &[Number]) -> Number {
    let b = t.borrow();
    b.lin[i].iter().map(|&(j, a)| a * x[j]).sum::<Number>() + b.lin_const[i]
}
