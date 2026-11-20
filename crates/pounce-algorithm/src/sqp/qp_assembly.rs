//! Build a [`pounce_qp::QpProblem`] from the NLP linearization at
//! the current SQP iterate `(x, λ_g)`.
//!
//! Standard SQP QP subproblem (Nocedal-Wright §18.1):
//!
//! ```text
//!     min  ½ pᵀ ∇²L(x, λ) p + ∇f(x)ᵀ p
//!     s.t.   bl_c ≤ c(x) + ∇c(x) p ≤ bu_c
//!            xl − x ≤ p ≤ xu − x
//! ```
//!
//! The QP's general bounds are shifted RHSs: `bl_qp = bl_c − c(x)`
//! and `bu_qp = bu_c − c(x)` (treating equalities as
//! `bl_c = bu_c = 0`). The QP's variable bounds are `xl − x` and
//! `xu − x`, so the QP primal `p` directly equals the SQP step.
//!
//! `SqpQpData` owns the sparse storage and exposes a borrowed
//! `QpProblem` view; this is the analog of
//! `pounce_qp::ElasticReformulation::as_qp`.
//!
//! # Low-rank Hessians (gh#1054)
//!
//! A limited-memory Hessian `B = σI + Σᵢ signᵢ vᵢ vᵢᵀ` is dense, and handing
//! it to the QP as a dense triplet costs `n²` storage and turns every sparse
//! factorization dense — at `n = 15 000` the first QP of a collocation model
//! did not finish. [`SqpQpData::build_low_rank`] instead *lifts* the QP with
//! one extra variable per term, `wᵢ = vᵢᵀ p`, held by an equality row:
//!
//! ```text
//!     min  ½ σ‖p‖² + ½ Σᵢ signᵢ wᵢ² + ∇f(x)ᵀ p
//!     s.t.   wᵢ − vᵢᵀ p = 0                      (r rows, always active)
//!            ... the original rows and bounds on p ...
//! ```
//!
//! Its Hessian is diagonal and its extra rows are `r` dense rows of
//! length `n`, so the KKT stays sparse; eliminating `w` gives back exactly
//! `½ pᵀ B p`. The lifted Hessian is indefinite (the `−1` terms) while its
//! restriction to the feasible set is `B`, positive definite, which is the
//! case the QP's inertia control is built for, so the problem is posed as
//! [`HessianInertia::Indefinite`]. The SQP never sees the lift:
//! [`SqpQpData::lift_working_set`] and [`SqpQpData::unlift_solution`] map
//! working sets and solutions across it at every QP call.
//!
//! # Stabilized subproblems (gh#1206)
//!
//! At a degenerate solution — linearly dependent active constraints (LICQ
//! fails) or weakly active ones (strict complementarity fails) — the KKT
//! matrix of the standard subproblem over the active set is singular, and an
//! active-set QP warm-started there has nothing it can factor. Stabilized SQP
//! (Wright 1998; Hager 1999; Fernández & Solodov 2010; Gill & Robinson 2013)
//! poses instead, at `(x_k, λ_k)` with a dual regularization `σ > 0`,
//!
//! ```text
//!     min  ∇fᵀp + ½ pᵀHp + λ_kᵀr + ‖r‖²/(2σ)
//!     s.t.   bl ≤ c + Jp − r ≤ bu,   xl ≤ x + p ≤ xu,   r free
//! ```
//!
//! Stationarity in `r` gives `r = σ(λ − λ_k)`, and eliminating `r` leaves the
//! KKT matrix `[H Jᵀ; J −σI]` on the active rows, nonsingular whatever the
//! rank of the active Jacobian. With `σ` of the order of the KKT residual it
//! converges superlinearly under second-order sufficiency alone.
//! [`SqpQpData::stabilize`] poses `r` as `m` extra free columns — `−I` in the
//! constraint matrix, `1/σ` on the Hessian diagonal, `λ_k` in the gradient —
//! through the same lift the low-rank form uses, so the rest of the SQP sees
//! the NLP's dimensions. And because `r` is free, the subproblem has a
//! trivially feasible point satisfying *any* working set exactly
//! ([`SqpQpData::stabilized_start`]), so the QP starts in its primal phase
//! with no pinned solve, no hint repair and no elastic phase 1.

use pounce_common::types::{Index, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{BoundStatus, ConsStatus, HessianInertia, QpProblem, QpSolution, WorkingSet};
use std::rc::Rc;

/// Owned linearization data for a single SQP iteration.
///
/// `n` / `m` are always the **NLP's** dimensions and `a` its Jacobian. When
/// the subproblem is lifted ([`Self::build_low_rank`]) the QP the solver
/// sees is larger: `h`, `g`, `bl`, `bu`, `xl`, `xu` hold the lifted
/// problem's data (lengths `n + r`, `m + r`), its first `n` / `m` entries
/// being the unlifted ones, and the lifted constraint matrix lives in the
/// private `lift`.
pub struct SqpQpData {
    pub n: usize,
    pub m: usize,

    pub h: SymTMatrix,
    pub g: Vec<Number>,
    pub a: GenTMatrix,
    pub bl: Vec<Number>,
    pub bu: Vec<Number>,
    pub xl: Vec<Number>,
    pub xu: Vec<Number>,
    pub hessian_inertia: HessianInertia,
    /// `max |∇²L_ij|` of the Hessian the step is computed against — the
    /// dense one, not the lifted diagonal, so the scale-relative QP
    /// tolerances do not depend on whether the QP was lifted.
    pub hess_inf: Number,
    lift: Option<Lift>,
    /// `σ` of a stabilized subproblem ([`Self::stabilize`]), `None` otherwise.
    stabilization: Option<Number>,
    /// The unstabilized Hessian, kept by [`Self::stabilize`] so the standard
    /// subproblem can still be posed ([`Self::as_standard_qp`]).
    standard_h: Option<SymTMatrix>,
}

/// The lifted constraint matrix of a low-rank or stabilized subproblem.
struct Lift {
    /// Number of lifted variables.
    extra_n: usize,
    /// Number of lifted rows (always equalities).
    extra_m: usize,
    /// Low rank: `[A 0; −Vᵀ I]`, `(m + r) × (n + r)`. Stabilized:
    /// `[A −I]`, `m × (n + m)`.
    a: GenTMatrix,
}

/// Sparse-triplet view of a derivative matrix. Indices are
/// 1-based per the pounce-linalg convention; values are owned.
#[derive(Clone)]
pub struct Triplet {
    pub n_rows: usize,
    pub n_cols: usize,
    pub irow: Vec<Index>,
    pub jcol: Vec<Index>,
    pub vals: Vec<Number>,
}

impl SqpQpData {
    /// Assemble from concrete linearization arrays at iterate `x`.
    ///
    /// * `grad_f` — `∇f(x)`, length `n`.
    /// * `c_vals` — `c(x)`, length `m` (may contain inequality
    ///   slack values too; convention is `bl_c[i] ≤ c[i] ≤ bu_c[i]`).
    /// * `bl_c`, `bu_c` — original NLP constraint bounds.
    /// * `xl_orig`, `xu_orig` — original NLP variable bounds.
    /// * `jac_c` — `∇c(x)` triplet, `m × n`.
    /// * `hess_lag` — `∇²L(x, λ_g)` triplet, `n × n` symmetric.
    pub fn build(
        x: &[Number],
        grad_f: &[Number],
        c_vals: &[Number],
        bl_c: &[Number],
        bu_c: &[Number],
        xl_orig: &[Number],
        xu_orig: &[Number],
        jac_c: Triplet,
        hess_lag: Triplet,
        hessian_inertia: HessianInertia,
    ) -> Self {
        let n = grad_f.len();
        let m = c_vals.len();
        assert_eq!(x.len(), n);
        assert_eq!(bl_c.len(), m);
        assert_eq!(bu_c.len(), m);
        assert_eq!(xl_orig.len(), n);
        assert_eq!(xu_orig.len(), n);
        assert_eq!(jac_c.n_rows, m);
        assert_eq!(jac_c.n_cols, n);
        assert_eq!(hess_lag.n_rows, n);
        assert_eq!(hess_lag.n_cols, n);

        let h_space = SymTMatrixSpace::new(n as Index, hess_lag.irow, hess_lag.jcol);
        let mut h = SymTMatrix::new(Rc::clone(&h_space));
        h.set_values(&hess_lag.vals);

        let a_space = GenTMatrixSpace::new(m as Index, n as Index, jac_c.irow, jac_c.jcol);
        let mut a = GenTMatrix::new(Rc::clone(&a_space));
        a.set_values(&jac_c.vals);

        // QP general bounds: bl_c − c(x), bu_c − c(x) — but
        // preserve ±∞ markers so the QP solver's one-sided
        // ratio test still treats them as unbounded.
        let mut bl = Vec::with_capacity(m);
        let mut bu = Vec::with_capacity(m);
        for i in 0..m {
            bl.push(shift_bound(bl_c[i], c_vals[i], true));
            bu.push(shift_bound(bu_c[i], c_vals[i], false));
        }

        // QP step bounds: xl_orig − x, xu_orig − x.
        let mut xl = Vec::with_capacity(n);
        let mut xu = Vec::with_capacity(n);
        for i in 0..n {
            xl.push(shift_bound(xl_orig[i], x[i], true));
            xu.push(shift_bound(xu_orig[i], x[i], false));
        }

        let hess_inf = h.values().iter().map(|v| v.abs()).fold(0.0, f64::max);
        Self {
            n,
            m,
            h,
            g: grad_f.to_vec(),
            a,
            bl,
            bu,
            xl,
            xu,
            hessian_inertia,
            hess_inf,
            lift: None,
            stabilization: None,
            standard_h: None,
        }
    }

    /// Assemble the subproblem for a low-rank Hessian `B = σI + Σ ±vvᵀ`
    /// without forming `B` — see the module docs. Same arguments as
    /// [`Self::build`] otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn build_low_rank(
        x: &[Number],
        grad_f: &[Number],
        c_vals: &[Number],
        bl_c: &[Number],
        bu_c: &[Number],
        xl_orig: &[Number],
        xu_orig: &[Number],
        jac_c: Triplet,
        b: &crate::sqp::lbfgs::LowRankHessian,
    ) -> Self {
        let n = grad_f.len();
        let m = c_vals.len();
        let r = b.vecs.len();
        let diag = Triplet {
            n_rows: n,
            n_cols: n,
            irow: (1..=n as Index).collect(),
            jcol: (1..=n as Index).collect(),
            vals: vec![b.sigma; n],
        };
        // The unlifted pieces, built exactly as `build` would.
        let mut out = Self::build(
            x,
            grad_f,
            c_vals,
            bl_c,
            bu_c,
            xl_orig,
            xu_orig,
            jac_c.clone(),
            diag,
            HessianInertia::Psd,
        );
        out.hess_inf = b.max_abs_entry();
        if r == 0 {
            // `B = σI` is already sparse; nothing to lift.
            return out;
        }

        let (nq, mq) = ((n + r) as Index, (m + r) as Index);
        // H = diag(σ I_n, sign_1 … sign_r).
        let h_space = SymTMatrixSpace::new(nq, (1..=nq).collect(), (1..=nq).collect());
        let mut h = SymTMatrix::new(h_space);
        let mut h_vals = vec![b.sigma; n];
        h_vals.extend_from_slice(&b.signs);
        h.set_values(&h_vals);

        // A = [J 0; −Vᵀ I]. Exact zeros of `v` are left out of the pattern.
        let mut irow = jac_c.irow;
        let mut jcol = jac_c.jcol;
        let mut vals = jac_c.vals;
        for (i, v) in b.vecs.iter().enumerate() {
            let row = (m + i + 1) as Index;
            for (j, &vj) in v.iter().enumerate() {
                if vj != 0.0 {
                    irow.push(row);
                    jcol.push((j + 1) as Index);
                    vals.push(-vj);
                }
            }
            irow.push(row);
            jcol.push((n + i + 1) as Index);
            vals.push(1.0);
        }
        let a_space = GenTMatrixSpace::new(mq, nq, irow, jcol);
        let mut a = GenTMatrix::new(a_space);
        a.set_values(&vals);

        out.h = h;
        out.g.resize(n + r, 0.0);
        out.bl.resize(m + r, 0.0);
        out.bu.resize(m + r, 0.0);
        out.xl.resize(n + r, NLP_LOWER_BOUND_INF);
        out.xu.resize(n + r, NLP_UPPER_BOUND_INF);
        out.hessian_inertia = HessianInertia::Indefinite;
        out.lift = Some(Lift {
            extra_n: r,
            extra_m: r,
            a,
        });
        out
    }

    /// Turn an unlifted subproblem into the stabilized one at multipliers
    /// `lambda_g` with dual regularization `sigma` — see the module docs.
    /// Returns `self` unchanged when it is already lifted (the low-rank form
    /// is not stabilized), when `m = 0`, or when `sigma` is not a positive
    /// finite number.
    pub fn stabilize(mut self, sigma: Number, lambda_g: &[Number]) -> Self {
        let (n, m) = (self.n, self.m);
        if self.lift.is_some() || m == 0 || !(sigma.is_finite() && sigma > 0.0) {
            return self;
        }
        debug_assert_eq!(lambda_g.len(), m);
        let nq = (n + m) as Index;
        // H = diag(H, I/σ): the original pattern plus one diagonal per `r_i`.
        let mut irow = self.h.irows().to_vec();
        let mut jcol = self.h.jcols().to_vec();
        let mut vals = self.h.values().to_vec();
        for i in 0..m {
            let k = (n + i + 1) as Index;
            irow.push(k);
            jcol.push(k);
            vals.push(1.0 / sigma);
        }
        let mut h = SymTMatrix::new(SymTMatrixSpace::new(nq, irow, jcol));
        h.set_values(&vals);
        // A = [J −I].
        let mut irow = self.a.irows().to_vec();
        let mut jcol = self.a.jcols().to_vec();
        let mut vals = self.a.values().to_vec();
        for i in 0..m {
            irow.push((i + 1) as Index);
            jcol.push((n + i + 1) as Index);
            vals.push(-1.0);
        }
        let mut a = GenTMatrix::new(GenTMatrixSpace::new(m as Index, nq, irow, jcol));
        a.set_values(&vals);

        self.standard_h = Some(std::mem::replace(&mut self.h, h));
        self.g.extend_from_slice(lambda_g);
        self.xl.resize(n + m, NLP_LOWER_BOUND_INF);
        self.xu.resize(n + m, NLP_UPPER_BOUND_INF);
        self.lift = Some(Lift {
            extra_n: m,
            extra_m: 0,
            a,
        });
        self.stabilization = Some(sigma);
        self
    }

    /// The standard (unstabilized) subproblem of a stabilized one, with the
    /// given general-row bounds; `None` when this subproblem is not
    /// stabilized. The fallback when a stabilized QP does not solve, and the
    /// only form that can report an infeasible linearization: the stabilized
    /// one is feasible by construction.
    pub fn as_standard_qp<'a>(
        &'a self,
        bl: &'a [Number],
        bu: &'a [Number],
    ) -> Option<QpProblem<'a>> {
        let h = self.standard_h.as_ref()?;
        Some(QpProblem {
            n: self.n,
            m: self.m,
            h,
            g: &self.g[..self.n],
            a: &self.a,
            bl,
            bu,
            xl: &self.xl[..self.n],
            xu: &self.xu[..self.n],
            hessian_inertia: self.hessian_inertia,
        })
    }

    /// `σ` when this subproblem is stabilized ([`Self::stabilize`]).
    pub fn stabilization(&self) -> Option<Number> {
        self.stabilization
    }

    /// A feasible point of the stabilized subproblem that satisfies the
    /// working set `w` (in the NLP's dimensions) exactly, as a warm start for
    /// [`pounce_qp::QpSolver::solve`]. `bl` / `bu` are the general-row bounds
    /// of the QP actually being solved — [`Self::bl`] / [`Self::bu`], or the
    /// second-order correction's re-centred ones. `None` when the subproblem
    /// is not stabilized.
    ///
    /// `p` sits on each active bound and at the projection of `0` onto its
    /// box elsewhere; then `r = Jp − t`, with `t` the active row bound, or
    /// `Jp` projected onto `[bl, bu]` for an inactive row. Every row and
    /// bound holds and every active one holds with equality, whatever `w`
    /// is — the free `r` absorbs what the linearization cannot meet — and it
    /// costs one product with `J`, no factorization.
    pub fn stabilized_start(
        &self,
        w: &WorkingSet,
        bl: &[Number],
        bu: &[Number],
    ) -> Option<pounce_qp::QpWarmStart> {
        self.stabilization?;
        let (n, m) = (self.n, self.m);
        let finite_lo = |v: Number| v > NLP_LOWER_BOUND_INF;
        let finite_up = |v: Number| v < NLP_UPPER_BOUND_INF;
        let mut bounds = Vec::with_capacity(n + m);
        let mut x = vec![0.0; n + m];
        for j in 0..n {
            let (lo, up) = (self.xl[j], self.xu[j]);
            let status = w.bounds.get(j).copied().unwrap_or(BoundStatus::Inactive);
            // An active status on an infinite bound cannot be honoured; the
            // variable is left free instead.
            let status = match status {
                _ if finite_lo(lo) && finite_up(up) && lo == up => BoundStatus::Fixed,
                BoundStatus::AtLower if !finite_lo(lo) => BoundStatus::Inactive,
                BoundStatus::AtUpper if !finite_up(up) => BoundStatus::Inactive,
                BoundStatus::Fixed if !(finite_lo(lo) && finite_up(up)) => BoundStatus::Inactive,
                s => s,
            };
            x[j] = match status {
                BoundStatus::AtLower | BoundStatus::Fixed => lo,
                BoundStatus::AtUpper => up,
                BoundStatus::Inactive => {
                    let mut v = 0.0;
                    if finite_lo(lo) && v < lo {
                        v = lo;
                    }
                    if finite_up(up) && v > up {
                        v = up;
                    }
                    v
                }
            };
            bounds.push(status);
        }
        bounds.resize(n + m, BoundStatus::Inactive);
        // J p over the NLP's own columns.
        let mut jp = vec![0.0; m];
        let (ai, aj, av) = (self.a.irows(), self.a.jcols(), self.a.values());
        for k in 0..av.len() {
            jp[(ai[k] - 1) as usize] += av[k] * x[(aj[k] - 1) as usize];
        }
        let mut constraints = Vec::with_capacity(m);
        for i in 0..m {
            let (lo, up) = (bl[i], bu[i]);
            let equality = finite_lo(lo) && finite_up(up) && lo == up;
            let status = if equality {
                ConsStatus::Equality
            } else {
                match w
                    .constraints
                    .get(i)
                    .copied()
                    .unwrap_or(ConsStatus::Inactive)
                {
                    ConsStatus::AtLower if finite_lo(lo) => ConsStatus::AtLower,
                    ConsStatus::AtUpper if finite_up(up) => ConsStatus::AtUpper,
                    // An `Equality` hint on a row that is not one, or an
                    // active status on an infinite side, is not honoured.
                    _ => ConsStatus::Inactive,
                }
            };
            let target = match status {
                ConsStatus::Equality | ConsStatus::AtLower => lo,
                ConsStatus::AtUpper => up,
                ConsStatus::Inactive => {
                    let mut t = jp[i];
                    if finite_lo(lo) && t < lo {
                        t = lo;
                    }
                    if finite_up(up) && t > up {
                        t = up;
                    }
                    t
                }
            };
            x[n + i] = jp[i] - target;
            constraints.push(status);
        }
        Some(pounce_qp::QpWarmStart {
            x,
            // `λ_k`, which `stabilize` stored as the residuals' gradient.
            lambda_g: self.g[n..n + m].to_vec(),
            lambda_x: vec![0.0; n + m],
            working: WorkingSet {
                bounds,
                constraints,
            },
        })
    }

    /// Borrowed `QpProblem` view ready for
    /// `pounce_qp::QpSolver::solve`.
    pub fn as_qp(&self) -> QpProblem<'_> {
        self.as_qp_with_row_bounds(&self.bl, &self.bu)
    }

    /// [`Self::as_qp`] with replacement general-row bounds (the
    /// second-order correction re-centres them). `bl` / `bu` must have
    /// [`Self::bl`]'s length, lifted rows included.
    pub fn as_qp_with_row_bounds<'a>(
        &'a self,
        bl: &'a [Number],
        bu: &'a [Number],
    ) -> QpProblem<'a> {
        let (en, em) = self
            .lift
            .as_ref()
            .map_or((0, 0), |l| (l.extra_n, l.extra_m));
        QpProblem {
            n: self.n + en,
            m: self.m + em,
            h: &self.h,
            g: &self.g,
            a: self.lift.as_ref().map_or(&self.a, |l| &l.a),
            bl,
            bu,
            xl: &self.xl,
            xu: &self.xu,
            hessian_inertia: self.hessian_inertia,
        }
    }

    /// Whether the QP the solver sees is lifted.
    pub fn is_lifted(&self) -> bool {
        self.lift.is_some()
    }

    /// A working set in the NLP's dimensions, extended for the lifted QP:
    /// the lifted rows are equalities, the lifted variables free.
    pub fn lift_working_set(&self, w: &WorkingSet) -> WorkingSet {
        let mut w = w.clone();
        if let Some(l) = &self.lift {
            w.bounds.truncate(self.n);
            w.constraints.truncate(self.m);
            w.bounds.resize(self.n + l.extra_n, BoundStatus::Inactive);
            w.constraints
                .resize(self.m + l.extra_m, ConsStatus::Equality);
        }
        w
    }

    /// A QP solution back in the NLP's dimensions: the lifted variables'
    /// values and the lifted rows' multipliers are dropped. Exact, not an
    /// approximation: the lifted rows only define `w`, so the step `p` and
    /// the multipliers of the original rows and bounds are those of the
    /// unlifted QP.
    pub fn unlift_solution(&self, mut sol: QpSolution) -> QpSolution {
        if self.lift.is_some() {
            sol.x.truncate(self.n);
            sol.lambda_g.truncate(self.m);
            sol.lambda_x.truncate(self.n);
            sol.working.bounds.truncate(self.n);
            sol.working.constraints.truncate(self.m);
            if let Some(ray) = sol.unbounded_ray.as_mut() {
                ray.truncate(self.n);
            }
        }
        sol
    }
}

/// Shift a bound by subtracting the current value. Preserves
/// `NLP_*_BOUND_INF` sentinels so the QP solver's `is_finite`
/// checks still trigger correctly.
fn shift_bound(bound: Number, current: Number, is_lower: bool) -> Number {
    if is_lower {
        if bound <= NLP_LOWER_BOUND_INF {
            NLP_LOWER_BOUND_INF
        } else {
            bound - current
        }
    } else if bound >= NLP_UPPER_BOUND_INF {
        NLP_UPPER_BOUND_INF
    } else {
        bound - current
    }
}
