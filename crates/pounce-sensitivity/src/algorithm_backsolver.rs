//! `PdSensBacksolver` — `SensBacksolver` adapter over the converged
//! `PdFullSpaceSolver` from `pounce-algorithm`.
//!
//! This is the Phase B.2 piece tracked in
//! [pounce#16](https://github.com/jkitchin/pounce/issues/16): it lets
//! `pounce-sensitivity` drive backsolves against the real converged
//! KKT factor, replacing the synthetic [`crate::DenseLuBacksolver`]
//! used by Phase B.1 unit tests.
//!
//! # Use
//!
//! 1. Register an `on_converged` callback on `IpoptApplication` via
//!    [`pounce_algorithm::application::IpoptApplication::set_on_converged`].
//! 2. Inside the callback, build a `PdSensBacksolver` from the four
//!    handles passed in (`data`, `cq`, `nlp`, `&mut pd_solver`).
//! 3. Hand it to [`crate::SensApplication`] / a `SensStepCalc` /
//!    [`crate::compute_reduced_hessian`] like any other
//!    [`SensBacksolver`].
//!
//! Upstream `SensSimpleBacksolver`
//! ([`ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.cpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.cpp))
//! is the analogous wrapper around `IpoptCalculatedQuantities` +
//! `PDSystemSolver` upstream.
//!
//! # Flat-slice ↔ `IteratesVector` mapping
//!
//! The full primal-dual state of pounce's IPM is the eight-block
//! compound `(x, s, λ_c, λ_d, z_l, z_u, v_l, v_u)` (see
//! [`pounce_algorithm::iterates_vector::IteratesVector`]). This
//! adapter packs / unpacks the flat slices that
//! [`crate::SensBacksolver`] takes as the concatenation
//! `x || s || λ_c || λ_d || z_l || z_u || v_l || v_u`, mirroring
//! upstream's `CompoundVector` layout (`IpCompoundVector.hpp`).
//!
//! # Reference
//!
//! Pirnay, H.; López-Negrete, R.; Biegler, L. T. (2012). *Optimal
//! sensitivity based on IPOPT*. Mathematical Programming Computation,
//! **4**(4), 307–331. DOI:
//! [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).
//! Verified via Crossref on 2026-05-13.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::iterates_vector::{IteratesVector, IteratesVectorMut};
use pounce_algorithm::kkt::pd_full_space_solver::{PdFullSpaceSolver, SigmaOverride};
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_vector::DenseVector;
use pounce_nlp::ipopt_nlp::IpoptNlp;

use crate::backsolver::SensBacksolver;

/// Adapter from `PdFullSpaceSolver` to [`SensBacksolver`]. Holds
/// owning clones of the four pieces of the algorithm's converged
/// state, plus the 8-block iterate template used to allocate fresh
/// RHS / LHS vectors.
///
/// The PD solver lives behind an `Rc<RefCell<…>>` because
/// [`SensBacksolver::solve`] is `&self` but the upstream signature
/// for `PdFullSpaceSolver::solve` is `&mut self` (it caches the
/// last-solve dependency tags and the augsys-improved flag). The
/// `RefCell` is single-thread-only, single-borrow, exactly matching
/// the call pattern from `pounce-sensitivity`'s pipeline.
///
/// Owning (rather than borrowing) the four handles is what lets a
/// `PdSensBacksolver` outlive the `on_converged` callback frame —
/// required by the public `Solver` session API in `pounce-algorithm`,
/// which retains the backsolver for repeated `parametric_step` /
/// `kkt_solve` / `compute_reduced_hessian` calls after the IPM has
/// returned. The data, cq, and nlp handles are already
/// `Rc<RefCell<…>>` cheap-clone handles upstream, so this carries no
/// allocation overhead.
#[derive(Clone)]
pub struct PdSensBacksolver {
    /// Shared, interior-mutable handle to the converged PD solver.
    /// Cloned from `PdSearchDirCalc::pd_solver_rc()` at construction.
    pd: Rc<RefCell<PdFullSpaceSolver>>,
    data: IpoptDataHandle,
    cq: IpoptCqHandle,
    nlp: Rc<RefCell<dyn IpoptNlp>>,
    /// Block dimensions in `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order.
    dims: [usize; 8],
    /// 8-block prototype used to mint fresh vectors with the same
    /// `VectorSpace`s as the converged iterate; cloned from
    /// `data.borrow().curr`.
    template: IteratesVector,
    /// Natural-units row/column scaling pair (pounce#128). The IPM's
    /// KKT factor is held in the NLP's internally **scaled** space
    /// (objective factor `df`, per-row constraint factors `dc` / `dd`
    /// from `nlp_scaling_method`; scaled multipliers `ỹ = (df/dc)·y`,
    /// `z̃ = df·z`, `ṽ = (df/dd)·v`, scaled slack `s̃ = dd·s`). The
    /// scaled 8-block primal-dual system is the two-sided diagonal
    /// scaling `K̃ = E K F` of the natural-units system, with
    /// per-block entries
    ///
    /// ```text
    ///        x      s        y_c      y_d     z_l/z_u   v_l/v_u
    /// E  =   df     df/dd_i  dc_i     dd_i    df        df
    /// F  =   1      1/dd_i   dc_i/df  dd_i/df 1/df      dd_r(j)/df
    /// ```
    ///
    /// (`dd_r(j)` = the d-row scaling of the j-th finite d-bound,
    /// through the `pd_l` / `pd_u` expansion). Hence
    /// `K⁻¹ = F K̃⁻¹ E`: scale the RHS by `E`, back-solve against the
    /// held factor, scale the result by `F`. Unlike a symmetric
    /// congruence this needs no square root, so it covers a negative
    /// `obj_scaling_factor` (maximization) and covers the z/v
    /// bound-multiplier rows exactly (those rows admit no symmetric
    /// diagonal: `K̃_{z,x} = df·Z·Pᵀ` but `K̃_{z,z} = X − x_L` is
    /// unscaled). `None` ⇔ scaling inactive, identity.
    ///
    /// **Variable scaling** (gh#486 stage 3) multiplies into the same
    /// pair. A change of variables `x̃ = d ⊙ x` contributes
    ///
    /// ```text
    ///        x      z_l/z_u          everything else
    /// E  =   1/d    1                1
    /// F  =   1/d    d_{px(j)}        1
    /// ```
    ///
    /// (`d_{px(j)}` = the factor of the variable carrying the j-th
    /// finite bound, through the `px_l` / `px_u` expansion). The `s`,
    /// `y_c`, `y_d`, `v_l` and `v_u` blocks are untouched because the
    /// substitution leaves `c`, `d` and their multipliers alone. The
    /// two contributions compose by elementwise product, in either
    /// order, because both are diagonal.
    conj: Option<Rc<ConjPair>>,
    /// Per-variable factors `d` the solve ran under (gh#486), in the
    /// algorithm's **var-x** space — i.e. already projected through
    /// the fixed-variable map, so entry `i` matches KKT row `i`.
    /// `None` ⇔ no variable scaling. Folded into [`Self::conj`] for
    /// the back-solves; kept here for the consumers that read the
    /// converged iterate and the model's matrices directly rather than
    /// through the factor (see [`crate::activity`]).
    d_var: Option<Rc<Vec<Number>>>,
    /// The same factors in the user TNLP's **full-x** space, the shape
    /// `finalize_solution_z_l` / `n_full_x`-length reports come in.
    /// `None` alongside [`Self::d_var`].
    d_full: Option<Rc<Vec<Number>>>,
    /// Var-x row of the variable each bound multiplier constrains,
    /// `z_l` entries then `z_u` entries, read off the `px_l` / `px_u`
    /// expansions. `None` when either expansion is not an
    /// `ExpansionMatrix` and the map cannot be recovered.
    bound_vars: Option<Rc<Vec<crate::backsolver::BoundRow>>>,
    /// The barrier geometry re-measured against the bounds the model
    /// declares, for a held iterate that came from crossover. `None` —
    /// meaning "read the calculated quantities as they stand" — on
    /// every solve that ended on an interior point. See
    /// [`DeclaredFrameBarrier`].
    declared: Option<Rc<DeclaredFrameBarrier>>,
    /// The barrier diagonals every sensitivity solve actually factors
    /// with: [`Self::declared`]'s pair when there is one, the
    /// calculated quantities otherwise, with [`sigma_pin_caps`]
    /// applied to both (gh#737). See [`EffectiveSigma`].
    sigma: EffectiveSigma,
}

/// The barrier diagonals the sensitivity path factors with, after both
/// corrections that stand between the calculated quantities and the
/// matrix: gh#654's choice of *frame*, and gh#737's ceiling on how
/// stiff a pin the frame is allowed to report.
///
/// `None` in a block means "the calculated quantity as it stands" —
/// no crossover frame to substitute and nothing over the ceiling — so
/// an ordinary solve still factors against the cached `Σ` object and
/// keeps the tag-keyed factorization cache warm.
#[derive(Clone)]
struct EffectiveSigma {
    /// `x`-block diagonal, or `None` for `cq.curr_sigma_x()`.
    x: Option<Rc<dyn pounce_linalg::Vector>>,
    /// `s`-block diagonal, or `None` for `cq.curr_sigma_s()`.
    s: Option<Rc<dyn pounce_linalg::Vector>>,
    /// The gh#737 ceiling per var-x row, `INFINITY` where none applies.
    /// Kept because a *released* solve rebuilds one variable's `Σ`
    /// entry from the bounds that stay active, and a rebuilt entry has
    /// to land under the same ceiling the rest of the diagonal did.
    cap_x: Rc<Vec<Number>>,
}

/// `Σ` and the active-bound slacks of a **crossed-over** iterate,
/// measured against the bounds the user declared rather than the ones
/// the barrier ran against (gh#654).
///
/// `bound_relax_factor` (default `1e-8`) widens every bound by `δ`
/// before the solve, and crossover (gh#612) then parks the iterate
/// exactly on the *declared* bound — a full `δ` inside the live relaxed
/// one. So the calculated quantities report a slack of exactly `δ` at
/// every active bound, where an interior iterate would have carried
/// `μ/z`, and the barrier diagonal `Σ = z/s` comes out as `z/δ` instead
/// of `z²/μ`. Since `δ` is capped at `constr_viol_tol` and `μ` ends near
/// `tol/(barrier_tol_factor+1)`, that is *looser* whenever `z·δ/μ > 1`,
/// which is the ordinary case.
///
/// `Σ` is the stiffness with which the barrier holds a bounded variable,
/// and a reduced Hessian read off the held factor carries a residual
/// error of exactly `O(1/Σ)` — the leftover of that pin being finite. So
/// the looser reading is a measurably less accurate covariance: on the
/// gh#654 fixture, `18x` at a bound multiplier of `4.5` and `396x` at
/// `994.5`, tracking `z·δ/μ` exactly.
///
/// The correction is the same one gh#646 applied to the reported
/// residuals: measure the crossed-over point in the frame it was solved
/// in. Nothing on the live iterate is touched — the relaxed bounds are
/// still what the algorithm ran against, and un-relaxing them after the
/// fact would mean replacing the NLP's bound `Rc`s and invalidating
/// every tag-keyed cache built on them. This is the consumer boundary
/// instead, where the question "how stiffly is this point held" is
/// actually being asked.
struct DeclaredFrameBarrier {
    /// Variable-bound contribution to the `x` diagonal.
    sigma_x: Rc<dyn pounce_linalg::Vector>,
    /// Inequality-row-bound contribution to the `s` diagonal. Relaxation
    /// widens `d_L` / `d_U` too, and crossover puts `s = d(x)` on the
    /// declared row bounds, so the row half of the defect is identical
    /// to the variable half.
    sigma_s: Rc<dyn pounce_linalg::Vector>,
    /// The declared-frame `x` slacks behind `sigma_x`, compressed in the
    /// `px_l` / `px_u` spaces. Kept because a *released* solve rebuilds
    /// one variable's `Σ` entry from the sides that stay active and has
    /// to rebuild it in the same frame.
    slack_x_l: Vec<Number>,
    slack_x_u: Vec<Number>,
}

/// Left/right diagonal pair for the natural-units back-solve; see the
/// `conj` field doc on [`PdSensBacksolver`]. Both vectors are
/// flat-KKT-length, in the `x‖s‖y_c‖y_d‖z_l‖z_u‖v_l‖v_u` packing.
struct ConjPair {
    /// `E`: multiplied into the RHS before the scaled-space solve.
    e: Vec<Number>,
    /// `F`: multiplied into the solution after the scaled-space solve.
    f: Vec<Number>,
}

impl PdSensBacksolver {
    /// The retained handles the activity classifier reads
    /// (crate-internal; see [`crate::activity`]).
    pub(crate) fn activity_handles(
        &self,
    ) -> (&IpoptDataHandle, &IpoptCqHandle, &Rc<RefCell<dyn IpoptNlp>>) {
        (&self.data, &self.cq, &self.nlp)
    }

    /// The barrier level the **reported point** sits on.
    ///
    /// Normally that is `IpoptData::curr_mu`: the solve stops on the
    /// `mu = 0` error with `mu` already driven to the floor, so the
    /// driver's last barrier parameter still describes the iterate it
    /// stopped at, and every complementarity product is within
    /// tolerance of it.
    ///
    /// It stops describing the iterate when a terminating path
    /// installs multipliers of its own. `ComputeFeasibilityMultipliers`
    /// (`IpIpoptAlg.cpp:893`, ported in gh#508) is the one that bites:
    /// on a square NLP -- `dim(x) == dim(y_c)`, so the objective is
    /// decorative and the answer is just the feasible point -- it zeroes
    /// all four bound-multiplier blocks, solves for the feasibility
    /// multipliers, and converges the check outright. A square problem
    /// can therefore be *reported solved at `mu = mu_init`* with every
    /// complementarity product identically zero, on iteration 1, having
    /// never reduced the barrier at all.
    ///
    /// Everything downstream that reads `curr_mu` then measures the
    /// point against a barrier it is not on. It is not cosmetic: the
    /// equation-11 barrier correction injects `mu` into the
    /// complementarity rows, and on such a point that term is pure
    /// error -- on the `cd_split_pin_mapping` fixture it flips the sign
    /// of the returned bound-multiplier step (`-1.004e-4` against a
    /// true `+1.004e-4`).
    ///
    /// Detected from the point, never from the problem's dimensions:
    /// bound rows present with every bound multiplier exactly zero is a
    /// state no barrier iterate can be in -- the algorithm holds
    /// `z > 0` strictly -- and is exactly what that path leaves behind.
    /// The barrier level there is `0`: the equation-11 correction has
    /// nothing to carry the step off, and the complementarity rows are
    /// already satisfied where they stand.
    ///
    /// The test is exact equality rather than a threshold on purpose.
    /// `curr_avrg_compl` was measured as the alternative and rejected:
    /// across the `pounce-sensitivity` suite it disagrees with
    /// `curr_mu` by up to 4.7x on ordinary terminations, so reading the
    /// barrier level off it would move every sensitivity result to fix
    /// one. The zeroing, by contrast, is assignment, not arithmetic.
    pub(crate) fn barrier_mu(&self) -> Number {
        let d = self.data.borrow();
        let mu = d.curr_mu;
        let Some(curr) = d.curr.as_ref() else {
            return mu;
        };
        let blocks = [&curr.z_l, &curr.z_u, &curr.v_l, &curr.v_u];
        let n_bound: Index = blocks.iter().map(|v| v.dim()).sum();
        if n_bound == 0 {
            // No bounds at all: there is no barrier either way, and the
            // complementarity blocks the caller would shift are empty.
            return mu;
        }
        if blocks.iter().all(|v| v.amax() == 0.0) {
            return 0.0;
        }
        mu
    }

    /// Construct from the four handles handed in by the `on_converged`
    /// callback. Errors if `data` has no `curr` (i.e. the algorithm
    /// never reached an iterate — should not happen on
    /// `SolveSucceeded`) or the NLP reports scaling data inconsistent
    /// with the converged iterate (see [`Self::natural_units_conj`]).
    pub fn new(
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        pd: Rc<RefCell<PdFullSpaceSolver>>,
    ) -> Result<Self, String> {
        let curr = data
            .borrow()
            .curr
            .clone()
            .ok_or_else(|| "no current iterate at convergence".to_string())?;
        let dims = [
            curr.x.dim() as usize,
            curr.s.dim() as usize,
            curr.y_c.dim() as usize,
            curr.y_d.dim() as usize,
            curr.z_l.dim() as usize,
            curr.z_u.dim() as usize,
            curr.v_l.dim() as usize,
            curr.v_u.dim() as usize,
        ];
        let (d_var, d_full) = Self::variable_factors(nlp, &dims)?;
        let conj = Self::natural_units_conj(nlp, &dims, d_var.as_ref().map(|v| v.as_slice()))?;
        let bound_vars = Self::bound_variable_rows(nlp, &dims);
        let declared = Self::declared_frame_barrier(data, nlp, &dims);
        let sigma = Self::effective_sigma(cq, declared.as_ref(), &dims);
        Ok(Self {
            pd,
            data: Rc::clone(data),
            cq: Rc::clone(cq),
            nlp: Rc::clone(nlp),
            dims,
            template: curr,
            conj,
            d_var,
            d_full,
            bound_vars,
            declared,
            sigma,
        })
    }

    /// Pick the frame (gh#654) and apply the ceiling (gh#737), once,
    /// at construction: both diagonals are functions of the converged
    /// state alone, and every solve has to factor against the same
    /// object for the factorization cache to hold.
    fn effective_sigma(
        cq: &IpoptCqHandle,
        declared: Option<&Rc<DeclaredFrameBarrier>>,
        dims: &[usize; 8],
    ) -> EffectiveSigma {
        let cap_x = Rc::new(sigma_pin_caps(cq, dims[0]));
        let base_x = match declared {
            Some(d) => Rc::clone(&d.sigma_x),
            None => cq.borrow().curr_sigma_x(),
        };
        let base_s = match declared {
            Some(d) => Rc::clone(&d.sigma_s),
            None => cq.borrow().curr_sigma_s(),
        };
        // The `s` block's single model coefficient is the `−I` that ties
        // each row's slack to its `d(x)` row, exactly `1` in the scaled
        // space this is measured in, so its ceiling is one scalar.
        let cap_s = sigma_pin_cap(1.0);
        let x = cap_sigma(&base_x, &|i| {
            cap_x.get(i).copied().unwrap_or(Number::INFINITY)
        })
        .or_else(|| declared.map(|d| Rc::clone(&d.sigma_x)));
        let s = cap_sigma(&base_s, &|_| cap_s).or_else(|| declared.map(|d| Rc::clone(&d.sigma_s)));
        EffectiveSigma { x, s, cap_x }
    }

    /// Re-measure `Σ` against the declared bounds when the held iterate
    /// came from crossover; `None` otherwise, and `None` whenever the
    /// NLP does not report its declared box (the fallback is then the
    /// calculated quantities, i.e. the pre-gh#654 behaviour).
    ///
    /// Deliberately gated on crossover rather than applied everywhere:
    /// an *interior* iterate is not near the declared bounds in any
    /// useful sense — it can sit up to `δ` **outside** one — and its
    /// `μ/z` standoff is the barrier's own geometry, which is the right
    /// thing to read. Only a purified point has the declared frame as
    /// its own.
    fn declared_frame_barrier(
        data: &IpoptDataHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Option<Rc<DeclaredFrameBarrier>> {
        let (curr, from_crossover) = {
            let d = data.borrow();
            (d.curr.clone()?, d.curr_from_crossover)
        };
        if !from_crossover {
            return None;
        }
        let nlp_ref = nlp.borrow();
        let (x_l, x_u) = nlp_ref.declared_x_bounds()?;
        let (d_l, d_u) = nlp_ref.declared_d_bounds()?;
        if x_l.len() != dims[4]
            || x_u.len() != dims[5]
            || d_l.len() != dims[6]
            || d_u.len() != dims[7]
        {
            return None;
        }
        let (sigma_x, slack_x_l, slack_x_u) = declared_frame_sigma(
            &*nlp_ref.px_l(),
            &*nlp_ref.px_u(),
            &*curr.x,
            &x_l,
            &x_u,
            &*curr.z_l,
            &*curr.z_u,
            dims[0],
        );
        let (sigma_s, _, _) = declared_frame_sigma(
            &*nlp_ref.pd_l(),
            &*nlp_ref.pd_u(),
            &*curr.s,
            &d_l,
            &d_u,
            &*curr.v_l,
            &*curr.v_u,
            dims[1],
        );
        Some(Rc::new(DeclaredFrameBarrier {
            sigma_x,
            sigma_s,
            slack_x_l,
            slack_x_u,
        }))
    }

    /// The barrier diagonals to factor with — [`Self::sigma`], which
    /// is the declared-frame pair when the held iterate came from
    /// crossover and the calculated quantities otherwise, either way
    /// under gh#737's ceiling. A block that needed neither correction
    /// stays `None`, which is what [`SigmaOverride::default`] carries
    /// and what leaves the cached diagonal in place.
    fn sigma_override(&self) -> SigmaOverride {
        SigmaOverride {
            x: self.sigma.x.clone(),
            s: self.sigma.s.clone(),
        }
    }

    /// Whether the held-factor back-solve may run
    /// `PdFullSpaceSolver`'s iterative refinement.
    ///
    /// Refinement iterates `x += K^-1 r` and measures `r` against the
    /// system it thinks it is solving. That is only the system this
    /// factor factors when no `SigmaOverride` is in play: after
    /// crossover (gh#654) the barrier diagonal is replaced with the
    /// declared-frame one, so the residual is taken against a matrix
    /// the held factor does not decompose, the loop cannot converge,
    /// and it escalates instead of improving anything. Same reason the
    /// release path (`solve_released_inner`) never refines.
    ///
    /// So the test is on the override itself and not on where it came
    /// from. It read `declared.is_none()` while crossover was the only
    /// thing that could produce one; gh#737's ceiling is a second, and
    /// it fires on an ordinary solve with `declared` empty. On the
    /// gh#737 fixture that combination returned `7.97e22` for a step of
    /// `-0.21` -- refinement escalating against the uncapped matrix,
    /// exactly the failure this predicate exists to prevent. Declared
    /// still implies an override, so this stays equivalent wherever it
    /// was already right.
    fn may_refine(&self) -> bool {
        self.sigma.x.is_none() && self.sigma.s.is_none()
    }

    /// The `x`-block barrier diagonal in the frame the held iterate
    /// belongs to, under the [`sigma_pin_caps`] ceiling. Crate-internal
    /// because [`crate::activity`] reads `Σ` straight off the iterate
    /// rather than through the factor, and the two must not disagree
    /// about which bounds the point is measured against or about how
    /// stiffly it is held there.
    pub(crate) fn barrier_sigma_x(&self) -> Rc<dyn pounce_linalg::Vector> {
        match self.sigma.x.as_ref() {
            Some(v) => Rc::clone(v),
            None => self.cq.borrow().curr_sigma_x(),
        }
    }

    /// [`Self::barrier_sigma_x`] for the `s` block.
    pub(crate) fn barrier_sigma_s(&self) -> Rc<dyn pounce_linalg::Vector> {
        match self.sigma.s.as_ref() {
            Some(v) => Rc::clone(v),
            None => self.cq.borrow().curr_sigma_s(),
        }
    }

    /// Shared body of the two released solves. `shift` moves the
    /// released multipliers onto their x rows, which the step needs and
    /// a Schur complement's unit vectors must not get.
    fn solve_released_inner(
        &self,
        released: &[usize],
        rhs: &[Number],
        lhs: &mut [Number],
        shift: bool,
    ) -> bool {
        if rhs.len() != self.dim() || lhs.len() != self.dim() {
            return false;
        }
        // Nothing released is an ordinary solve. Taking this early lets
        // callers route every solve through here without paying a
        // re-factorization for a step that releases nothing.
        if released.is_empty() {
            return self.solve(rhs, lhs);
        }
        let Some(sigma) = self.released_sigma_x(released) else {
            return false;
        };
        self.solve_released_prebuilt(released, sigma, rhs, lhs, shift)
    }

    /// [`Self::solve_released_inner`] with the released `Σ` supplied by
    /// the caller. Repeated solves against ONE released operator must
    /// pass the same `Rc` every time: the factorization cache keys on
    /// the sigma object's tag, so a sigma rebuilt per call forces a
    /// re-factorization per call, while a held one factorizes once and
    /// back-solves thereafter.
    pub(crate) fn solve_released_prebuilt(
        &self,
        released: &[usize],
        sigma: Rc<dyn pounce_linalg::Vector>,
        rhs: &[Number],
        lhs: &mut [Number],
        shift: bool,
    ) -> bool {
        if rhs.len() != self.dim() || lhs.len() != self.dim() {
            return false;
        }
        let mut scaled: Vec<Number> = match self.conj.as_ref() {
            Some(c) => rhs.iter().zip(c.e.iter()).map(|(&r, &e)| r * e).collect(),
            None => rhs.to_vec(),
        };
        // A released bound's multiplier is fixed at zero, so its row
        // has no equation and the right-hand side there is meaningless.
        // It is also dangerous: the elimination folds a multiplier
        // row's entry in as r_z / s, and at a tightly active bound the
        // barrier-correction term every parametric right-hand side
        // carries, -mu on each bound row, folds to -mu / s = -z by
        // complementarity, an order-one injection into the released
        // variable's equation. Measured on a two-variable QP that bent
        // the released direction from the analytic [1.227, 0.454] to
        // [1.154, 0.194].
        for &r in released {
            if r < scaled.len() {
                scaled[r] = 0.0;
            }
        }
        if shift && !self.shift_released_rhs(released, &mut scaled) {
            return false;
        }
        if !self.solve_released_scaled(sigma, &scaled, lhs) {
            return false;
        }
        if let Some(c) = self.conj.as_ref() {
            for (l, &f) in lhs.iter_mut().zip(c.f.iter()) {
                *l *= f;
            }
        }
        true
    }

    /// The barrier's `x` diagonal with each released bound's own `z / s`
    /// taken off the variable it constrains -- subtracted rather than
    /// zeroed, since a variable bounded on both sides contributes twice
    /// and only one side is being released.
    ///
    /// Both the starting diagonal and the slacks the surviving sides are
    /// rebuilt from come from the frame the held iterate belongs to
    /// (gh#654): mixing a declared-frame `Σ` with relaxed-frame slacks
    /// would leave the released variable pinned in one frame and its
    /// neighbours in the other.
    pub(crate) fn released_sigma_x(
        &self,
        released: &[usize],
    ) -> Option<Rc<dyn pounce_linalg::Vector>> {
        self.active_set_sigma_x(released, &[])
    }

    /// [`Self::released_sigma_x`], also raising the diagonal on
    /// variables a step brings onto a bound.
    ///
    /// The two directions an active set can move are the same
    /// modification to one diagonal. A bound that leaves has its
    /// `z / s` taken off the variable it constrains, and a bound that
    /// becomes active has one put on, so a single vector describes
    /// both and a single factorization serves the whole correction.
    /// The two arguments are in different index spaces and both are
    /// `usize`: `released` holds compound KKT rows of bound
    /// multipliers, `pinned` holds var-x rows. Passing one where the
    /// other belongs is not a type error and will not be caught here.
    pub(crate) fn active_set_sigma_x(
        &self,
        released: &[usize],
        pinned: &[(usize, Number)],
    ) -> Option<Rc<dyn pounce_linalg::Vector>> {
        use pounce_linalg::dense_vector::DenseVectorSpace;
        let rows = self.bound_vars.as_deref()?;
        let base_row = rows.first()?.row;
        let dense = |v: Rc<dyn pounce_linalg::Vector>| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|d| d.expanded_values())
        };
        let mut sigma = dense(self.barrier_sigma_x())?;
        let (slack_l, slack_u) = match self.declared.as_ref() {
            Some(d) => (d.slack_x_l.clone(), d.slack_x_u.clone()),
            None => {
                let cq_ref = self.cq.borrow();
                let l = dense(cq_ref.curr_slack_x_l())?;
                let u = dense(cq_ref.curr_slack_x_u())?;
                (l, u)
            }
        };
        let (z_l, z_u) = {
            let d = self.data.borrow();
            let curr = d.curr.as_ref()?;
            (dense(Rc::clone(&curr.z_l))?, dense(Rc::clone(&curr.z_u))?)
        };
        // Rebuild each released variable's entry from its bounds that
        // stay active, rather than subtracting the released bound's
        // `z / s` from the cached total. The subtraction differences
        // two numbers of order `z / s`, 1e7 and up at a tightly
        // active bound, and its correctness rests on the cache having
        // built its total from bitwise-identical products. Rebuilding
        // makes the released side an exact zero by construction and
        // depends on nothing about the cache.
        for &r in released {
            let br = rows.iter().find(|b| b.row == r)?;
            let mut fresh = 0.0;
            for other in rows.iter().filter(|b| b.var_row == br.var_row) {
                if released.contains(&other.row) {
                    continue;
                }
                let k = other.row - base_row - if other.lower { 0 } else { self.dims[4] };
                let (z, s) = if other.lower {
                    (*z_l.get(k)?, *slack_l.get(k)?)
                } else {
                    (*z_u.get(k)?, *slack_u.get(k)?)
                };
                if s == 0.0 || !s.is_finite() {
                    return None;
                }
                fresh += z / s;
            }
            // Under the same ceiling the rest of the diagonal is held
            // to (gh#737): a rebuilt entry is `z/s` off the iterate
            // like any other, and a released variable is the one most
            // likely to be reached through a constraint row.
            let cap = self
                .sigma
                .cap_x
                .get(br.var_row)
                .copied()
                .unwrap_or(Number::INFINITY);
            *sigma.get_mut(br.var_row)? = fresh.min(cap);
        }
        for &(var_row, add) in pinned {
            let cap = self
                .sigma
                .cap_x
                .get(var_row)
                .copied()
                .unwrap_or(Number::INFINITY);
            let slot = sigma.get_mut(var_row)?;
            *slot = pinned_entry(*slot, add, cap);
        }
        let space = DenseVectorSpace::new(sigma.len() as Index);
        let mut out = DenseVector::new(space);
        out.values_mut().copy_from_slice(&sigma);
        Some(Rc::new(out) as Rc<dyn pounce_linalg::Vector>)
    }

    /// Zero the released multipliers' own rows of a scaled-space
    /// right-hand side and move each multiplier onto its variable's x
    /// row.
    ///
    /// Dropping `sigma` gives the released *matrix*; the released
    /// right-hand side still wants the multiplier moved across. The
    /// elimination folds a multiplier row in as `r_z / s`, so zeroing
    /// that row and adding the multiplier to the x row directly reaches
    /// the same place without `s` appearing at all -- which is the
    /// whole point of re-factoring rather than downdating.
    fn shift_released_rhs(&self, released: &[usize], rhs: &mut [Number]) -> bool {
        let Some(rows) = self.bound_vars.as_deref() else {
            return false;
        };
        let Some(base_row) = rows.first().map(|b| b.row) else {
            return false;
        };
        let d = self.data.borrow();
        let Some(curr) = d.curr.as_ref() else {
            return false;
        };
        let dense = |v: &Rc<dyn pounce_linalg::Vector>| -> Option<Vec<Number>> {
            v.as_any()
                .downcast_ref::<DenseVector>()
                .map(|x| x.expanded_values())
        };
        let (Some(z_l), Some(z_u)) = (dense(&curr.z_l), dense(&curr.z_u)) else {
            return false;
        };
        for &r in released {
            let Some(br) = rows.iter().find(|b| b.row == r) else {
                return false;
            };
            let k = r - base_row - if br.lower { 0 } else { self.dims[4] };
            let Some(&z) = (if br.lower { z_l.get(k) } else { z_u.get(k) }) else {
                return false;
            };
            if r >= rhs.len() || br.var_row >= rhs.len() {
                return false;
            }
            rhs[r] = 0.0;
            // the sign the x row carries this side's multiplier with
            rhs[br.var_row] += if br.lower { -z } else { z };
        }
        true
    }

    /// One back-solve against the re-factored system, in the solver's
    /// internal scaled space.
    fn solve_released_scaled(
        &self,
        sigma: Rc<dyn pounce_linalg::Vector>,
        rhs: &[Number],
        lhs: &mut [Number],
    ) -> bool {
        let off = self.offsets();
        let rhs_mut0 = self.template.make_new_zeroed();
        let mut rhs_iv = rhs_mut0.freeze();
        let mut res_iv = self.template.make_new_zeroed();
        if !(write_rhs_block(&mut rhs_iv.x, &rhs[off[0]..off[1]])
            && write_rhs_block(&mut rhs_iv.s, &rhs[off[1]..off[2]])
            && write_rhs_block(&mut rhs_iv.y_c, &rhs[off[2]..off[3]])
            && write_rhs_block(&mut rhs_iv.y_d, &rhs[off[3]..off[4]])
            && write_rhs_block(&mut rhs_iv.z_l, &rhs[off[4]..off[5]])
            && write_rhs_block(&mut rhs_iv.z_u, &rhs[off[5]..off[6]])
            && write_rhs_block(&mut rhs_iv.v_l, &rhs[off[6]..off[7]])
            && write_rhs_block(&mut rhs_iv.v_u, &rhs[off[7]..off[8]]))
        {
            return false;
        }
        if !self.pd.borrow_mut().solve_with_sigma(
            &self.data,
            &self.cq,
            &self.nlp,
            1.0,
            0.0,
            &rhs_iv,
            &mut res_iv,
            // NOT refined: the release path asks the held factor
            // for the solution of a *different* system (one bound
            // released), so the refinement loop would measure a
            // residual against a matrix this factor does not factor,
            // stagnate, and escalate. See `solve_scaled_space` for
            // why the ordinary back-solves do refine.
            /* allow_inexact = */
            true,
            /* improve_solution = */ false,
            SigmaOverride {
                x: Some(sigma),
                // The release is an x-block operation; the row block
                // keeps whichever frame the held iterate belongs to.
                s: self.sigma_override().s,
            },
        ) {
            return false;
        }
        read_res_block(&*res_iv.x, &mut lhs[off[0]..off[1]])
            && read_res_block(&*res_iv.s, &mut lhs[off[1]..off[2]])
            && read_res_block(&*res_iv.y_c, &mut lhs[off[2]..off[3]])
            && read_res_block(&*res_iv.y_d, &mut lhs[off[3]..off[4]])
            && read_res_block(&*res_iv.z_l, &mut lhs[off[4]..off[5]])
            && read_res_block(&*res_iv.z_u, &mut lhs[off[5]..off[6]])
            && read_res_block(&*res_iv.v_l, &mut lhs[off[6]..off[7]])
            && read_res_block(&*res_iv.v_u, &mut lhs[off[7]..off[8]])
    }

    /// Var-x row behind each `z_l` then `z_u` entry, through the
    /// `px_l` / `px_u` expansions. `None` when either is not an
    /// `ExpansionMatrix` or reports the wrong length -- the release
    /// half then stays off rather than guessing a mapping.
    fn bound_variable_rows(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Option<Rc<Vec<crate::backsolver::BoundRow>>> {
        let nlp_ref = nlp.borrow();
        let z_l_off = dims[0] + dims[1] + dims[2] + dims[3];
        let mut out = Vec::with_capacity(dims[4] + dims[5]);
        for (pm, n_v, off, lower) in [
            (nlp_ref.px_l(), dims[4], z_l_off, true),
            (nlp_ref.px_u(), dims[5], z_l_off + dims[4], false),
        ] {
            if n_v == 0 {
                continue;
            }
            let em = pm
                .as_any()
                .downcast_ref::<pounce_linalg::expansion_matrix::ExpansionMatrix>()?;
            let pos = em.expanded_pos_indices();
            if pos.len() != n_v {
                return None;
            }
            for (k, &p) in pos.iter().enumerate() {
                let p = p as usize;
                if p >= dims[0] {
                    return None;
                }
                out.push(crate::backsolver::BoundRow {
                    row: off + k,
                    var_row: p,
                    lower,
                });
            }
        }
        Some(Rc::new(out))
    }

    /// Read the variable factors the solve ran under off the NLP and
    /// project them into the algorithm's var-x space (gh#486 stage 3).
    ///
    /// Returns `(None, None)` when no variable scaling is active.
    /// Errors when the reported vector does not match the NLP's own
    /// full-x width, or when the projection does not fill the `x`
    /// block: either would silently mis-pair a factor with a variable,
    /// which is the whole failure mode this plumbing exists to avoid.
    #[allow(clippy::type_complexity)]
    fn variable_factors(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
    ) -> Result<(Option<Rc<Vec<Number>>>, Option<Rc<Vec<Number>>>), String> {
        let nlp_ref = nlp.borrow();
        let Some(d_full) = nlp_ref.variable_scaling() else {
            return Ok((None, None));
        };
        let n_full = nlp_ref.n_full_x() as usize;
        if d_full.len() != n_full {
            return Err(format!(
                "variable scaling length {} != n_full_x {}",
                d_full.len(),
                n_full
            ));
        }
        // NaN is not a "no-op" factor and neither is zero; the wrapper
        // refuses both at setup, so seeing one here means the vector
        // did not come from the wrapper that ran.
        if let Some(bad) = d_full.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(format!(
                "variable scaling factor {bad} is not finite and positive"
            ));
        }
        let mut d_var = vec![Number::NAN; dims[0]];
        for (full, &factor) in d_full.iter().enumerate() {
            if let Some(var) = nlp_ref.full_x_to_var_x(full as Index) {
                let slot = d_var.get_mut(var as usize).ok_or_else(|| {
                    format!("var-x index {var} outside x block of width {}", dims[0])
                })?;
                *slot = factor;
            }
        }
        if let Some(pos) = d_var.iter().position(|v| v.is_nan()) {
            return Err(format!(
                "variable scaling left var-x column {pos} of {} unmapped",
                dims[0]
            ));
        }
        Ok((Some(Rc::new(d_var)), Some(Rc::new(d_full))))
    }

    /// The per-variable factors the held solve ran under, in the
    /// algorithm's **var-x** space (one entry per `x`-block KKT row),
    /// or `None` when no variable scaling was active (gh#486).
    pub fn variable_scaling(&self) -> Option<&[Number]> {
        self.d_var.as_deref().map(|v| v.as_slice())
    }

    /// [`Self::variable_scaling`] in the user TNLP's **full-x** space:
    /// the shape of an `n_full_x`-length report, with the columns the
    /// solve dropped as fixed still present.
    pub fn variable_scaling_full(&self) -> Option<&[Number]> {
        self.d_full.as_deref().map(|v| v.as_slice())
    }

    /// Build the natural-units scaling pair `(E, F)` from the NLP's
    /// effective scaling and the variable factors `d_var` the solve
    /// ran under (see the field doc on [`Self::conj`]).
    /// Returns `Ok(None)` when no scaling is active. Errors when the
    /// NLP reports scaling data inconsistent with the converged
    /// iterate's block dimensions (would silently corrupt every
    /// back-solve) or a zero/non-finite `df`.
    fn natural_units_conj(
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        dims: &[usize; 8],
        d_var: Option<&[Number]>,
    ) -> Result<Option<Rc<ConjPair>>, String> {
        let nlp_ref = nlp.borrow();
        let df = nlp_ref.obj_scaling_factor();
        let dc = nlp_ref.c_scale_vec();
        let dd = nlp_ref.d_scale_vec();
        // `d_var` counts as active scaling on its own: a solve with
        // unit objective and row factors but a change of variables
        // still holds its factor in scaled coordinates.
        if df == 1.0 && dc.is_none() && dd.is_none() && d_var.is_none() {
            return Ok(None);
        }
        // df may be negative (obj_scaling_factor < 0 means maximize);
        // the two-sided scaling needs no square root, only df ≠ 0.
        if !df.is_finite() || df == 0.0 {
            return Err(format!("invalid obj_scaling_factor {df}"));
        }
        if let Some(v) = &dc {
            if v.len() != dims[2] {
                return Err(format!("c_scale length {} != y_c dim {}", v.len(), dims[2]));
            }
        }
        if let Some(v) = &dd {
            if v.len() != dims[3] || dims[1] != dims[3] {
                return Err(format!(
                    "d_scale length {} != y_d dim {} (s dim {})",
                    v.len(),
                    dims[3],
                    dims[1]
                ));
            }
        }
        if let Some(d) = d_var {
            if d.len() != dims[0] {
                return Err(format!(
                    "variable scaling length {} != x dim {}",
                    d.len(),
                    dims[0]
                ));
            }
        }
        // Per-entry source scale for a compressed bound-multiplier
        // block: entry j of z_l / v_l covers the row
        // `px_l.expanded_pos[j]` / `pd_l.expanded_pos[j]` of `src`.
        // Used for the v blocks (source `d_scale`, indexed by
        // inequality row) and the z blocks (source `d_var`, indexed by
        // var-x column).
        let bound_row_scale = |pm: Rc<dyn pounce_linalg::matrix::Matrix>,
                               src: Option<&[Number]>,
                               n_v: usize,
                               which: &str|
         -> Result<Vec<Number>, String> {
            let Some(vals) = src else {
                return Ok(vec![1.0; n_v]);
            };
            if n_v == 0 {
                return Ok(Vec::new());
            }
            let Some(em) = pm
                .as_any()
                .downcast_ref::<pounce_linalg::expansion_matrix::ExpansionMatrix>()
            else {
                return Err(format!("{which} is not an ExpansionMatrix"));
            };
            let pos = em.expanded_pos_indices();
            if pos.len() != n_v {
                return Err(format!(
                    "{which} expansion length {} != {} block dim {}",
                    pos.len(),
                    which,
                    n_v
                ));
            }
            pos.iter()
                .map(|&r| {
                    vals.get(r as usize).copied().ok_or_else(|| {
                        format!(
                            "{which} expansion row {r} out of scale-vector range {}",
                            vals.len()
                        )
                    })
                })
                .collect()
        };
        let vl_dd = bound_row_scale(nlp_ref.pd_l(), dd.as_deref(), dims[6], "pd_l")?;
        let vu_dd = bound_row_scale(nlp_ref.pd_u(), dd.as_deref(), dims[7], "pd_u")?;
        // The variable factor carried by each finite x-bound, through
        // the same expansion (gh#486). `d_var` indexes var-x columns,
        // and `px_l` / `px_u` say which column each z entry belongs to.
        let zl_dx = bound_row_scale(nlp_ref.px_l(), d_var, dims[4], "px_l")?;
        let zu_dx = bound_row_scale(nlp_ref.px_u(), d_var, dims[5], "px_u")?;
        drop(nlp_ref);

        let total: usize = dims.iter().sum();
        let mut e = Vec::with_capacity(total);
        let mut f = Vec::with_capacity(total);
        // x block: E = df/d_i, F = 1/d_i. `df` is the objective scale;
        // the `1/d_i` on both sides is the change of variables —
        // `∇f̃ = ∇f ⊘ d` puts the RHS in scaled units and `x = x̃ ⊘ d`
        // brings the solution back.
        match d_var {
            Some(d) => {
                e.extend(d.iter().map(|&di| df / di));
                f.extend(d.iter().map(|&di| 1.0 / di));
            }
            None => {
                e.extend(std::iter::repeat_n(df, dims[0]));
                f.extend(std::iter::repeat_n(1.0, dims[0]));
            }
        }
        // s block: E = df/dd_i, F = 1/dd_i (slacks live in scaled d-space).
        match &dd {
            Some(v) => {
                e.extend(v.iter().map(|&ddi| df / ddi));
                f.extend(v.iter().map(|&ddi| 1.0 / ddi));
            }
            None => {
                e.extend(std::iter::repeat_n(df, dims[1]));
                f.extend(std::iter::repeat_n(1.0, dims[1]));
            }
        }
        // y_c block: E = dc_i, F = dc_i/df.
        match &dc {
            Some(v) => {
                e.extend(v.iter().copied());
                f.extend(v.iter().map(|&dci| dci / df));
            }
            None => {
                e.extend(std::iter::repeat_n(1.0, dims[2]));
                f.extend(std::iter::repeat_n(1.0 / df, dims[2]));
            }
        }
        // y_d block: E = dd_i, F = dd_i/df.
        match &dd {
            Some(v) => {
                e.extend(v.iter().copied());
                f.extend(v.iter().map(|&ddi| ddi / df));
            }
            None => {
                e.extend(std::iter::repeat_n(1.0, dims[3]));
                f.extend(std::iter::repeat_n(1.0 / df, dims[3]));
            }
        }
        // z_l / z_u blocks: E = df, F = d_{px(j)}/df (z̃ = (df/d)·z,
        // and the slack diagonal x̃ − x̃_L = d·(x − x_L) carries the
        // variable factor, so the two cancel in the row and leave it
        // identical to the natural one — E takes no `d` at all).
        // Without variable scaling this is the pre-#486 `F = 1/df`:
        // bounds on x are unscaled and the slack diagonal is shared.
        e.extend(std::iter::repeat_n(df, dims[4] + dims[5]));
        f.extend(zl_dx.iter().map(|&dx| dx / df));
        f.extend(zu_dx.iter().map(|&dx| dx / df));
        // v_l / v_u blocks: E = df, F = dd_r/df (ṽ = (df/dd)·v and the
        // slack diagonal s̃ − d̃_l = dd·(s − d_l) carries the d-row
        // scale).
        e.extend(std::iter::repeat_n(df, dims[6] + dims[7]));
        f.extend(vl_dd.iter().map(|&ddr| ddr / df));
        f.extend(vu_dd.iter().map(|&ddr| ddr / df));
        Ok(Some(Rc::new(ConjPair { e, f })))
    }

    /// Effective objective scaling factor `df` of the converged NLP
    /// (1.0 when no scaling is active).
    pub fn obj_scaling_factor(&self) -> Number {
        self.nlp.borrow().obj_scaling_factor()
    }

    /// Effective NLP scaling at convergence:
    /// `(obj_scaling_factor, c_scale, d_scale)`. The vectors are
    /// `None` when the corresponding block carries no row scaling.
    pub fn nlp_scaling(&self) -> (Number, Option<Vec<Number>>, Option<Vec<Number>>) {
        let n = self.nlp.borrow();
        (n.obj_scaling_factor(), n.c_scale_vec(), n.d_scale_vec())
    }

    /// Inertia-correction perturbations `(δ_x, δ_s, δ_c, δ_d)` baked
    /// into the held KKT factor (the IPM's `current_perturbation`
    /// state at convergence). All zero ⇔ the final factorization was
    /// unregularized and the natural-units back-solves invert the
    /// exact KKT matrix. Nonzero ⇔ the factor carries a (scaled-space)
    /// regularization, so sensitivity outputs — covariance in
    /// particular — are perturbed and no longer exactly
    /// scaling-invariant; consumers should check this before trusting
    /// `-inv(reduced_hessian)` on ill-conditioned problems
    /// (pounce#128 follow-up).
    pub fn kkt_perturbations(&self) -> [Number; 4] {
        let p = self.data.borrow().perturbations;
        [p.delta_x, p.delta_s, p.delta_c, p.delta_d]
    }

    /// Map user-facing 0-based `g(x)` indices of parameter-pin
    /// equality constraints to flat KKT rows **and** the pin rows'
    /// `dc_i` scaling factors, in one pass. The KKT row of pin `i` is
    /// `n_x + n_s + c_block_idx`, i.e. the matching `y_c` slot, found
    /// through `IpoptNlp::full_g_to_c_block` so the c/d split's row
    /// permutation is honored (pounce#128 follow-up: the previous
    /// direct `n_x + n_s + g_idx` mapping silently picked wrong rows
    /// when inequalities preceded the pins). The scales are 1.0 when
    /// no constraint scaling is active; they relate the natural and
    /// solver-space reduced Hessians via
    /// `H̃_ij = (df / (dc_i·dc_j)) · H_ij`. Errors when a pin index
    /// is out of range or refers to an inequality row.
    pub fn pin_rows_and_c_scales(
        &self,
        pin_g_indices: &[Index],
    ) -> Result<(Vec<Index>, Vec<Number>), String> {
        let y_c_offset = (self.dims[0] + self.dims[1]) as Index;
        let nlp = self.nlp.borrow();
        let dc = nlp.c_scale_vec();
        let n_full_g = nlp.n_full_g();
        let mut rows = Vec::with_capacity(pin_g_indices.len());
        let mut scales = Vec::with_capacity(pin_g_indices.len());
        for &gi in pin_g_indices {
            // n_full_g() defaults to 0 for IpoptNlp impls that don't
            // report it; only range-check when it's meaningful.
            if gi < 0 || (n_full_g > 0 && gi >= n_full_g) {
                return Err(format!(
                    "pin constraint index {gi} out of range [0, m={n_full_g})"
                ));
            }
            let Some(ci) = nlp.full_g_to_c_block(gi) else {
                return Err(format!(
                    "pin constraint index {gi} is an inequality (not an equality row); \
                     parameter pins must be exact equalities"
                ));
            };
            rows.push(y_c_offset + ci);
            scales.push(dc.as_ref().map(|v| v[ci as usize]).unwrap_or(1.0));
        }
        Ok((rows, scales))
    }

    /// KKT-row half of [`Self::pin_rows_and_c_scales`].
    pub fn map_pin_g_to_kkt_rows(&self, pin_g_indices: &[Index]) -> Result<Vec<Index>, String> {
        Ok(self.pin_rows_and_c_scales(pin_g_indices)?.0)
    }

    /// Scaling half of [`Self::pin_rows_and_c_scales`].
    pub fn pin_c_scales(&self, pin_g_indices: &[Index]) -> Result<Vec<Number>, String> {
        Ok(self.pin_rows_and_c_scales(pin_g_indices)?.1)
    }

    /// Block dimensions of the compound KKT vector at convergence, in
    /// `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)` order. Sum equals
    /// [`SensBacksolver::dim`]. Useful when a caller needs to compute
    /// the flat offset of a non-x block (e.g. `n_x + n_s` for the
    /// start of the equality-multiplier `y_c` block).
    pub fn block_dims(&self) -> [usize; 8] {
        self.dims
    }

    /// Map a 0-based **full-g** index (user-TNLP `g(x)` order) to its
    /// 0-based position in the equality-multiplier `y_c` block, or
    /// `None` when the constraint is an inequality (it lives in the `d`
    /// block, not `y_c`). Delegates to the held NLP's c/d-split map.
    ///
    /// Pin-row construction must route through this: the flat KKT row of
    /// a pinned equality is `n_x + n_s + full_g_to_c_block(g)`, NOT
    /// `n_x + n_s + g` — those differ whenever any inequality precedes
    /// the pinned equality in `g(x)`.
    pub fn full_g_to_c_block(&self, full_idx: Index) -> Option<Index> {
        self.nlp.borrow().full_g_to_c_block(full_idx)
    }

    /// Map a 0-based **full-x** index (user-TNLP variable order) to its
    /// 0-based position in the algorithm-side `x` block, or `None` when
    /// the solve removed the column because `x_l == x_u` under
    /// `fixed_variable_treatment = make_parameter`. Delegates to the
    /// held NLP's fixed-variable map.
    ///
    /// The `x` counterpart of [`Self::full_g_to_c_block`], and it must
    /// be routed through for the same reason: the flat KKT row of a
    /// user variable is `full_x_to_var_x(i)`, NOT `i` — those differ
    /// whenever any fixed variable precedes it in the user's `x`.
    /// Reports and iterates are in full-x, the factor is in var-x, and
    /// nothing about the two spaces is distinguishable by length alone
    /// on a model that happens to have no fixed variables.
    pub fn full_x_to_var_x(&self, full_idx: Index) -> Option<Index> {
        self.nlp.borrow().full_x_to_var_x(full_idx)
    }

    /// The user TNLP's variable count, the domain of
    /// [`Self::full_x_to_var_x`]. Distinct from the `x` block width
    /// whenever the solve removed a fixed variable.
    pub fn n_full_x(&self) -> Index {
        self.nlp.borrow().n_full_x()
    }

    /// The user TNLP's constraint count, the `g` counterpart of
    /// [`Self::n_full_x`]: the length of a full-g report and the
    /// domain of [`Self::full_g_to_c_block`]. Distinct from either
    /// KKT row block's width, since the c/d split sends equalities to
    /// `y_c` and inequalities to `s`/`y_d`.
    pub fn n_full_g(&self) -> Index {
        self.nlp.borrow().n_full_g()
    }

    /// `E` itself, the vector [`Self::solve`] pre-multiplies its
    /// right-hand side by.
    ///
    /// The counterpart of [`SensBacksolver::natural_units_factor`],
    /// which reports `F`. A caller holding a residual it assembled from
    /// the algorithm's own calculated quantities holds it in the scaled
    /// frame, and `solve` wants its right-hand side in natural units:
    /// `K̃ = E K F` with `v_scaled = F⁻¹ v_nat` gives
    /// `r_scaled = E r_nat`, so that caller divides by this before
    /// handing the residual over. Passing `r_scaled` straight in applies
    /// `E` twice and leaves a diagonally mis-scaled Newton direction.
    pub(crate) fn scaled_rhs_factor(&self) -> Option<&[Number]> {
        self.conj.as_ref().map(|c| c.e.as_slice())
    }

    /// [`Self::offsets`], for the corrector's residual assembly, which
    /// writes one calculated-quantity block at a time.
    pub(crate) fn offsets_public(&self) -> [usize; 9] {
        self.offsets()
    }

    /// [`Self::pack`], for the corrector, which builds a trial iterate
    /// from the flat point it is stepping.
    pub(crate) fn pack_public(&self, flat: &[Number]) -> Result<IteratesVectorMut, ()> {
        self.pack(flat)
    }

    /// The converged iterate, flattened into the compound layout.
    ///
    /// The corrector steps a point rather than a step, so it needs the
    /// iterate the step is measured from, in the same layout the step
    /// arrives in.
    pub(crate) fn curr_flat(&self, out: &mut [Number]) -> Result<(), ()> {
        if out.len() != self.dim() {
            return Err(());
        }
        let curr = {
            let d = self.data.borrow();
            d.curr.clone().ok_or(())?
        };
        let off = self.offsets();
        let blocks: [&Rc<dyn pounce_linalg::vector::Vector>; 8] = [
            &curr.x, &curr.s, &curr.y_c, &curr.y_d, &curr.z_l, &curr.z_u, &curr.v_l, &curr.v_u,
        ];
        for (i, b) in blocks.iter().enumerate() {
            let vals = crate::vec_util::dense_to_vec(&***b);
            let (a, e) = (off[i], off[i + 1]);
            if vals.len() != e - a {
                return Err(());
            }
            out[a..e].copy_from_slice(&vals);
        }
        Ok(())
    }

    /// Cumulative block offsets: `offset(i)` is the start index of
    /// block `i` in the flat slice.
    fn offsets(&self) -> [usize; 9] {
        let mut o = [0usize; 9];
        for i in 0..8 {
            o[i + 1] = o[i] + self.dims[i];
        }
        o
    }

    /// Pack a flat slice into a freshly-allocated `IteratesVectorMut`
    /// shaped like the converged iterate.
    fn pack(&self, flat: &[Number]) -> Result<IteratesVectorMut, ()> {
        let mut out = self.template.make_new_zeroed();
        let off = self.offsets();
        let blocks: [&mut Box<dyn pounce_linalg::vector::Vector>; 8] = [
            &mut out.x,
            &mut out.s,
            &mut out.y_c,
            &mut out.y_d,
            &mut out.z_l,
            &mut out.z_u,
            &mut out.v_l,
            &mut out.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let slice = &flat[off[i]..off[i + 1]];
            let dv = blk.as_any_mut().downcast_mut::<DenseVector>().ok_or(())?;
            dv.set_values(slice);
        }
        Ok(out)
    }

    /// Read an `IteratesVectorMut` into a flat slice. Uses
    /// [`DenseVector::expanded_values`] rather than `values()` so
    /// blocks that the IPM left in homogeneous-scalar form (typical
    /// for empty z_l/z_u/v_l/v_u when the TNLP has no bounds) are
    /// materialized rather than panicking.
    fn unpack(&self, iv: &IteratesVectorMut, out: &mut [Number]) -> Result<(), ()> {
        let off = self.offsets();
        let blocks: [&Box<dyn pounce_linalg::vector::Vector>; 8] = [
            &iv.x, &iv.s, &iv.y_c, &iv.y_d, &iv.z_l, &iv.z_u, &iv.v_l, &iv.v_u,
        ];
        for (i, blk) in blocks.into_iter().enumerate() {
            let dst = &mut out[off[i]..off[i + 1]];
            if dst.is_empty() {
                continue;
            }
            let dv = (**blk).as_any().downcast_ref::<DenseVector>().ok_or(())?;
            let ev = dv.expanded_values();
            dst.copy_from_slice(&ev);
        }
        Ok(())
    }
}

impl PdSensBacksolver {
    /// Batched-RHS back-solve over the held factor. `rhs_flat` and
    /// `lhs_flat` are row-major `(n_rhs, dim)` buffers. Equivalent to
    /// looping [`SensBacksolver::solve`] over each row but reuses one
    /// frozen `IteratesVector` for the RHS and one `IteratesVectorMut`
    /// for the result across all `n_rhs` calls into
    /// [`PdFullSpaceSolver::solve`]. The pack step writes into the
    /// existing `DenseVector` storage via `Rc::get_mut` +
    /// `set_values`, and the unpack step reads it back via `values()`
    /// /`scalar()` — skipping the per-call 8-block `make_new_zeroed`
    /// (Box alloc) in `pack` and the per-block `expanded_values()` Vec
    /// alloc in `unpack` that otherwise dominate the held-factor
    /// back-solve cost under `jax.jacrev` over a JaxProblem solve
    /// (pounce#77 follow-up).
    ///
    /// The matrix and perturbation state inside `PdFullSpaceSolver`
    /// are unchanged across calls, so each iteration hits the cached
    /// fast path in `solve_once` (`uptodate && !pretend_singular`).
    ///
    /// Like [`SensBacksolver::solve`], results are in **natural
    /// (unscaled) units** — see [`Self::solve_many_scaled_space`] for
    /// the raw solver-space back-solve.
    pub fn solve_many(&self, rhs_flat: &[Number], lhs_flat: &mut [Number], n_rhs: usize) -> bool {
        match &self.conj {
            None => self.solve_many_scaled_space(rhs_flat, lhs_flat, n_rhs),
            Some(c) => {
                let total = self.dim();
                if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
                    return false;
                }
                let mut rhs_scaled = rhs_flat.to_vec();
                for row in rhs_scaled.chunks_mut(total) {
                    for (r, &ei) in row.iter_mut().zip(c.e.iter()) {
                        *r *= ei;
                    }
                }
                if !self.solve_many_scaled_space(&rhs_scaled, lhs_flat, n_rhs) {
                    return false;
                }
                for row in lhs_flat.chunks_mut(total) {
                    for (l, &fi) in row.iter_mut().zip(c.f.iter()) {
                        *l *= fi;
                    }
                }
                true
            }
        }
    }

    /// Batched-RHS back-solve against the held factor in the solver's
    /// internal **scaled** space (no natural-units conjugation). Same
    /// buffer contract as [`Self::solve_many`].
    pub fn solve_many_scaled_space(
        &self,
        rhs_flat: &[Number],
        lhs_flat: &mut [Number],
        n_rhs: usize,
    ) -> bool {
        let total = self.dim();
        if rhs_flat.len() != n_rhs * total || lhs_flat.len() != n_rhs * total {
            return false;
        }
        if n_rhs == 0 {
            return true;
        }
        let off = self.offsets();

        // Both cached tiers below assemble their elimination from the
        // *calculated* `Σ` and slacks, and fire against whatever factor
        // the last solve left behind. Neither is the right system when
        // the held iterate came from crossover and the declared-frame
        // diagonal is in force (gh#654), and the tag check cannot be
        // relied on to decline: on the first call after convergence the
        // cached tags are the algorithm's own final solve, which used
        // exactly the diagonal being corrected. So skip them outright
        // and take the per-RHS path, which carries the override. That
        // costs the batched path its inlining on a crossed-over solve —
        // one factorization, then a back-substitution per RHS, against
        // the tag cache that the shared override vector keeps warm.
        let sigma = self.sigma_override();
        // Refined, for the reason spelled out in `solve_scaled_space`:
        // one shot, no outer loop.
        let allow_inexact = !self.may_refine();
        let corrected = self.declared.is_some();

        // Tier 1: fully-inline flat-slice path. `PdFullSpaceSolver::
        // solve_many_cached_flat` downcasts the slack / z / v vectors to
        // `DenseVector` and the bound-expansion matrices to
        // `ExpansionMatrix` once at the top, then runs Phase 1 / Phase 3
        // as raw scatter-add / divide loops on flat slices with no dyn
        // dispatch in the per-RHS inner loops. Returns `None` if a
        // downcast fails (homogeneous-on-non-empty block, unusual matrix
        // type) — we fall to Tier 2.
        if !corrected {
            let mut pd_ref = self.pd.borrow_mut();
            let fast_flat = pd_ref.solve_many_cached_flat(
                &self.data, &self.cq, &self.nlp, n_rhs, rhs_flat, lhs_flat, self.dims,
            );
            match fast_flat {
                Some(true) => return true,
                Some(false) => return false,
                None => { /* fall through to Tier 2 */ }
            }
        }

        // Tier 2: closure-based cached-factor path. Same single
        // back-substitution through the linsol, but Phase 1 / Phase 3
        // go through `dyn Vector` / `dyn Matrix` ops on a per-RHS
        // `IteratesVectorMut`. Slower than Tier 1 but correct for
        // homogeneous DenseVectors and non-`ExpansionMatrix` bound
        // expansions.
        if !corrected {
            let mut pd_ref = self.pd.borrow_mut();
            let fast = pd_ref.solve_many_cached(
                &self.data,
                &self.cq,
                &self.nlp,
                n_rhs,
                |k, iv| {
                    let row = &rhs_flat[k * total..(k + 1) * total];
                    let _ = write_rhs_box(&mut iv.x, &row[off[0]..off[1]])
                        && write_rhs_box(&mut iv.s, &row[off[1]..off[2]])
                        && write_rhs_box(&mut iv.y_c, &row[off[2]..off[3]])
                        && write_rhs_box(&mut iv.y_d, &row[off[3]..off[4]])
                        && write_rhs_box(&mut iv.z_l, &row[off[4]..off[5]])
                        && write_rhs_box(&mut iv.z_u, &row[off[5]..off[6]])
                        && write_rhs_box(&mut iv.v_l, &row[off[6]..off[7]])
                        && write_rhs_box(&mut iv.v_u, &row[off[7]..off[8]]);
                },
                |k, iv| {
                    let row = &mut lhs_flat[k * total..(k + 1) * total];
                    let _ = read_res_block(&*iv.x, &mut row[off[0]..off[1]])
                        && read_res_block(&*iv.s, &mut row[off[1]..off[2]])
                        && read_res_block(&*iv.y_c, &mut row[off[2]..off[3]])
                        && read_res_block(&*iv.y_d, &mut row[off[3]..off[4]])
                        && read_res_block(&*iv.z_l, &mut row[off[4]..off[5]])
                        && read_res_block(&*iv.z_u, &mut row[off[5]..off[6]])
                        && read_res_block(&*iv.v_l, &mut row[off[6]..off[7]])
                        && read_res_block(&*iv.v_u, &mut row[off[7]..off[8]]);
                },
            );
            match fast {
                Some(true) => return true,
                Some(false) => return false,
                None => { /* fall through to per-RHS loop */ }
            }
        }

        // Per-RHS fallback: reuse one frozen rhs and one mut sol across
        // all n_rhs `solve` calls.
        let rhs_mut0 = self.template.make_new_zeroed();
        let mut rhs_iv = rhs_mut0.freeze();
        let mut res_iv = self.template.make_new_zeroed();

        let mut pd_ref = self.pd.borrow_mut();
        for k in 0..n_rhs {
            let rhs_row = &rhs_flat[k * total..(k + 1) * total];
            let lhs_row = &mut lhs_flat[k * total..(k + 1) * total];

            if !write_rhs_block(&mut rhs_iv.x, &rhs_row[off[0]..off[1]])
                || !write_rhs_block(&mut rhs_iv.s, &rhs_row[off[1]..off[2]])
                || !write_rhs_block(&mut rhs_iv.y_c, &rhs_row[off[2]..off[3]])
                || !write_rhs_block(&mut rhs_iv.y_d, &rhs_row[off[3]..off[4]])
                || !write_rhs_block(&mut rhs_iv.z_l, &rhs_row[off[4]..off[5]])
                || !write_rhs_block(&mut rhs_iv.z_u, &rhs_row[off[5]..off[6]])
                || !write_rhs_block(&mut rhs_iv.v_l, &rhs_row[off[6]..off[7]])
                || !write_rhs_block(&mut rhs_iv.v_u, &rhs_row[off[7]..off[8]])
            {
                return false;
            }

            let ok = pd_ref.solve_with_sigma(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                allow_inexact,
                /* improve_solution = */ false,
                sigma.clone(),
            );
            if !ok {
                return false;
            }

            if !read_res_block(&*res_iv.x, &mut lhs_row[off[0]..off[1]])
                || !read_res_block(&*res_iv.s, &mut lhs_row[off[1]..off[2]])
                || !read_res_block(&*res_iv.y_c, &mut lhs_row[off[2]..off[3]])
                || !read_res_block(&*res_iv.y_d, &mut lhs_row[off[3]..off[4]])
                || !read_res_block(&*res_iv.z_l, &mut lhs_row[off[4]..off[5]])
                || !read_res_block(&*res_iv.z_u, &mut lhs_row[off[5]..off[6]])
                || !read_res_block(&*res_iv.v_l, &mut lhs_row[off[6]..off[7]])
                || !read_res_block(&*res_iv.v_u, &mut lhs_row[off[7]..off[8]])
            {
                return false;
            }
        }
        true
    }
}

/// Headroom on the representability half of [`declared_slack_floor`],
/// matching `ipopt_cq`'s `SIGMA_OVERFLOW_HEADROOM` (gh#655) because it
/// bounds the same quantity for the same reason: `Σ_x` sums a lower and
/// an upper ratio into one diagonal entry, so bounding each by `MAX/4`
/// bounds the sum by `MAX/2` with room for the rounding in the divide.
const SIGMA_OVERFLOW_HEADROOM: Number = 4.0;

/// The smallest offset from a bound that the declared-frame slack is
/// allowed to be: the larger of what double precision tells apart from
/// the bound itself, and what keeps `Σ = z/s` inside the double range.
///
/// A crossed-over point is *on* its active bounds, so its declared-frame
/// slack is the residual of the QP step and the line search — measured
/// around `1.8e-12` (gh#653), comfortably above both, so this does
/// nothing on the ordinary path. Each half covers a different corner:
///
/// * **`eps·max(1,|bound|)`** — a pivot landing on the bound exactly.
///   `Σ = z/0` is not a stiffer pin, it is a `NaN` in the KKT matrix.
///   Flooring says the honest thing instead: below this distance the
///   point *is* the bound, and the slack carries no further information
///   about how far inside it sits.
/// * **`z_max/(f64::MAX/4)`** — a multiplier large enough that `z/s`
///   leaves the double range even at a slack this frame considers
///   resolvable. This is gh#655's floor, which the live path gained in
///   `CalculateSafeSlack`; the declared frame does not go through that
///   function, so without carrying the bound here explicitly the
///   guarantee would stop at the frame boundary. It takes `max_i z_i`
///   over the block rather than each bound's own `z`, matching gh#655
///   and conservative in the harmless direction — raising a slack only
///   lowers `Σ`. A non-finite `z` leaves the floor alone; the iterate
///   finiteness checks own that case.
///
/// What is deliberately *not* borrowed is the rest of
/// `CalculateSafeSlack`: it raises a below-floor slack to
/// `max(μ/z, s_min)`, i.e. straight back to the `μ/z` standoff crossover
/// exists to remove (the same reason gh#646 declined it for the residual
/// report). Only the representability bound crosses over, because that
/// one is about what a double can hold, not about where the barrier
/// would have put the point.
fn declared_slack_floor(bound: Number, z_max: Number) -> Number {
    let resolvable = Number::EPSILON * bound.abs().max(1.0);
    if z_max.is_finite() && z_max > 0.0 {
        // Divide before multiplying so a `z_max` near `f64::MAX` cannot
        // overflow the floor itself.
        resolvable.max(z_max / (Number::MAX / SIGMA_OVERFLOW_HEADROOM))
    } else {
        resolvable
    }
}

/// Headroom on [`sigma_pin_caps`]' representability bound: how far
/// above one ulp the surviving Schur contribution `a²/Σ` is required to
/// stay.
///
/// The bound itself is where that contribution reaches exactly one ulp
/// of unity, which is the edge rather than a safe distance from it, and
/// the edge is where measurement puts the failure: on the gh#737
/// fixture `Σ = 3.6e22` still returns the exact step and `6.9e27`
/// returns none of it, while the issue's own bracket runs from `7.1e14`
/// correct to `3.4e23` zero. Backing off is nearly free — the ceiling
/// only ever binds on an entry that would have been capped anyway, and
/// dropping it from `1/eps` to `1/(64·eps)` moves the pin's residual
/// leak from `2e-16` to `1.4e-14`, still an order under the roundoff a
/// back-solve on any real model carries. The cost of the other
/// direction is the defect surviving between the ceiling and the
/// failure.
const SIGMA_PIN_HEADROOM: Number = 64.0;

/// The largest barrier diagonal each variable may carry and still be
/// reachable through the constraint rows it appears in, one entry per
/// var-x row; `INFINITY` for a variable in no constraint row, and an
/// all-`INFINITY` vector when the Jacobians are not triplet matrices
/// and the column magnitudes cannot be read.
///
/// # What the ceiling is
///
/// `Σ_i` sits on the diagonal of KKT row `i`, alongside that variable's
/// Jacobian entries `a_ji` in the constraint columns. Eliminating the
/// variable through its own diagonal leaves each constraint row `j`
/// holding `a_ji²/Σ_i` — the whole of what row `j` still knows about
/// variable `i`. Once that quantity falls below the roundoff of the
/// row it lands in, the constraint is no longer represented: the
/// factorization sees a row it cannot pivot on, and what comes back is
/// whatever the singularity handling substitutes.
///
/// Requiring `a²/Σ` to stay at or above one ulp is therefore the
/// ceiling `Σ_i ≤ a_i²/eps`, with `a_i` the largest of the variable's
/// constraint coefficients, and [`SIGMA_PIN_HEADROOM`] backing off from
/// the edge. The quadratic form is the scale-invariant one: a change of
/// variables `x_i → c·x_i` sends `Σ_i → Σ_i/c²` and `a_ji → a_ji/c`, so
/// the ceiling tracks the diagonal it bounds. Both quantities are read
/// in the solver's own scaled space, which is the space the factor
/// lives in.
///
/// # What it is not
///
/// It is not a release. A capped bound is still a bound, and still
/// pinned about as hard as double precision expresses: at the ceiling
/// the variable moves by `eps·SIGMA_PIN_HEADROOM/a²` per unit of force,
/// which for a constraint coefficient of order one is roundoff.
/// Zeroing the entry
/// instead would let a genuinely held variable off its bound entirely
/// and answer a different question. The rule is only that a pin the
/// matrix cannot represent is not a stiffer pin — the same argument
/// [`declared_slack_floor`] makes one step further down, where `z/0` is
/// not an infinitely stiff pin but a `NaN`.
///
/// A variable in no constraint row is left alone: there is no row for
/// its diagonal to swamp, and that is the gh#653 / gh#654 case, where a
/// bound-pinned variable coupled to the rest of the model only through
/// the Hessian wants every digit of stiffness it has.
fn sigma_pin_caps(cq: &IpoptCqHandle, n_x: usize) -> Vec<Number> {
    use pounce_linalg::triplet::GenTMatrix;

    let mut a_max: Vec<Number> = vec![0.0; n_x];
    let (jac_c, jac_d) = {
        let c = cq.borrow();
        (c.curr_jac_c(), c.curr_jac_d())
    };
    for jac in [jac_c, jac_d] {
        let Some(t) = jac.as_any().downcast_ref::<GenTMatrix>() else {
            return vec![Number::INFINITY; n_x];
        };
        for (&col, &v) in t.jcols().iter().zip(t.values().iter()) {
            let j = (col - 1) as usize;
            if let Some(slot) = a_max.get_mut(j) {
                *slot = slot.max(v.abs());
            }
        }
    }
    a_max
        .into_iter()
        .map(|a| {
            if a > 0.0 && a.is_finite() {
                sigma_pin_cap(a)
            } else {
                Number::INFINITY
            }
        })
        .collect()
}

/// [`sigma_pin_caps`]' ceiling for one coefficient magnitude, `a²/eps`
/// backed off by [`SIGMA_PIN_HEADROOM`].
///
/// Grouped as two divides so a large `a` overflows to `INFINITY` — no
/// ceiling, which is right, since no representable `Σ` reaches it — in
/// place of a spurious finite product. The floor at the other end is
/// the one that matters: for a coefficient small enough that the
/// ceiling underflows, *no* positive `Σ` keeps that coefficient
/// representable, and a ceiling at or below `1` would not be a looser
/// pin but a released bound. There is nothing to buy there, so those
/// report no ceiling too and leave the diagonal as it stands.
fn sigma_pin_cap(a: Number) -> Number {
    let cap = (a / Number::EPSILON) * (a / SIGMA_PIN_HEADROOM);
    if cap > 1.0 { cap } else { Number::INFINITY }
}

/// One diagonal entry after a step brings a bound onto its variable:
/// the entry it had, plus the newly active bound's contribution, held
/// under that variable's ceiling.
///
/// The corrector's pinned contribution (gh#733) is `mu / s²` off the
/// slack the *endpoint* has, landing on a variable the corrector has
/// just decided sits on a bound -- gh#737's own case, reached through
/// a second door. The ceiling is a property of the entry rather than
/// of where the entry came from, so it applies to the sum and not to
/// the addend: two contributions that are individually representable
/// can still swamp the row together.
///
/// # How far the addend can actually reach
///
/// `pinned_rows` itself bounds that slack from below by nothing beyond
/// `> 0`, but the caller does: `correct_step` clamps the iterate to
/// `margin = 1e-10 * (1 + |base_i|)` inside each bound *before*
/// measuring it. So the addend is bounded after all, at
/// `mu / margin²`, and at a converged `mu` of `1e-9` on a variable of
/// order one that is `2.5e10` -- three orders under the `7.0e13` a
/// unit Jacobian coefficient allows.
///
/// The ceiling therefore binds on this path only where the coefficient
/// is small enough to bring it down to meet the addend, below
/// `sqrt(mu · eps · SIGMA_PIN_HEADROOM) / margin`, about `2e-2` at
/// that `mu`. That is measured, not deduced: on PR #738 the corrector
/// was driven with pinned coefficients of `1e-3` and `1e-4` and two of
/// twelve cases moved -- an iteration count of 1 against 12 at
/// identical residuals, and a residual differing in the fifth digit.
/// Live, and far too small to assert on.
///
/// So this clamp is prophylaxis rather than a fix for a reachable
/// failure: what makes the corrector's door narrow is a `1e-10` margin
/// in a different file, which no rule keeps in step with this ceiling.
/// The ceiling costs nothing to hold here and does not depend on that
/// margin staying where it is.
fn pinned_entry(had: Number, add: Number, cap: Number) -> Number {
    (had + add).min(cap)
}

/// `min(Σ, cap)` entrywise, or the input unchanged (and `None`) when no
/// entry is over its ceiling.
///
/// Returning `None` rather than an equal copy is what keeps an ordinary
/// solve factoring against the object the calculated quantities already
/// cache: the factorization cache keys on the `Σ` object's tag, and a
/// fresh vector per construction would cost a re-factorization for a
/// diagonal that is bit-identical.
fn cap_sigma(
    sigma: &Rc<dyn pounce_linalg::Vector>,
    cap: &dyn Fn(usize) -> Number,
) -> Option<Rc<dyn pounce_linalg::Vector>> {
    use pounce_linalg::dense_vector::DenseVectorSpace;

    let dv = sigma.as_any().downcast_ref::<DenseVector>()?;
    let mut vals = dv.expanded_values();
    let mut hit = false;
    for (i, v) in vals.iter_mut().enumerate() {
        let c = cap(i);
        if *v > c {
            *v = c;
            hit = true;
        }
    }
    if !hit {
        return None;
    }
    let mut out = DenseVector::new(DenseVectorSpace::new(vals.len() as Index));
    out.values_mut().copy_from_slice(&vals);
    Some(Rc::new(out) as Rc<dyn pounce_linalg::Vector>)
}

/// `Σ = Σ_l z_l/s_l + Σ_u z_u/s_u` for one primal block, with the slacks
/// taken against `b_l` / `b_u` instead of the NLP's live (relaxed)
/// bounds. Returns the diagonal plus the two slack vectors it was built
/// from, compressed in the `p_l` / `p_u` spaces.
///
/// Mirrors `IpoptCalculatedQuantities`' own `curr_sigma_x` /
/// `curr_sigma_s` — the same `Pᵀx − b` slack and the same
/// `add_m_sinv_z` accumulation — so the only difference between this and
/// the cached value is which bounds it measured against, and the floor
/// above in place of the safe-slack correction.
#[allow(clippy::too_many_arguments)]
fn declared_frame_sigma(
    p_l: &dyn pounce_linalg::Matrix,
    p_u: &dyn pounce_linalg::Matrix,
    primal: &dyn pounce_linalg::Vector,
    b_l: &[Number],
    b_u: &[Number],
    z_l: &dyn pounce_linalg::Vector,
    z_u: &dyn pounce_linalg::Vector,
    n: usize,
) -> (Rc<dyn pounce_linalg::Vector>, Vec<Number>, Vec<Number>) {
    use pounce_linalg::Vector;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    // One scalar over both sides of the block, as gh#655 does: the two
    // ratios land in the same `Σ` entry, so the bound that keeps their
    // sum representable has to be taken over both.
    let z_max = z_l.amax().max(z_u.amax());

    // `lower`: s = Pᵀx − b_l. Otherwise: s = b_u − Pᵀx.
    let slack = |p: &dyn pounce_linalg::Matrix, b: &[Number], lower: bool| -> DenseVector {
        let mut v = DenseVector::new(DenseVectorSpace::new(b.len() as Index));
        if !b.is_empty() {
            v.values_mut().copy_from_slice(b);
        }
        let (alpha, beta) = if lower { (1.0, -1.0) } else { (-1.0, 1.0) };
        p.trans_mult_vector(alpha, primal, beta, &mut v);
        for (s, &bi) in v.values_mut().iter_mut().zip(b.iter()) {
            *s = s.max(declared_slack_floor(bi, z_max));
        }
        v
    };
    let s_l = slack(p_l, b_l, true);
    let s_u = slack(p_u, b_u, false);

    let mut sigma = DenseVector::new(DenseVectorSpace::new(n as Index));
    sigma.set(0.0);
    p_l.add_m_sinv_z(1.0, &s_l, z_l, &mut sigma);
    p_u.add_m_sinv_z(1.0, &s_u, z_u, &mut sigma);
    (
        Rc::new(sigma) as Rc<dyn pounce_linalg::Vector>,
        s_l.expanded_values(),
        s_u.expanded_values(),
    )
}

/// Write `slice` into the `DenseVector` behind `b` in place. Used by
/// the fast path's `write_rhs` closure, where the new
/// `PdFullSpaceSolver::solve_many_cached` API hands back an
/// `IteratesVectorMut` (Box-backed blocks).
fn write_rhs_box(b: &mut Box<dyn pounce_linalg::vector::Vector>, slice: &[Number]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(dv) = b.as_any_mut().downcast_mut::<DenseVector>() else {
        return false;
    };
    dv.set_values(slice);
    true
}

/// Write `slice` into the `DenseVector` behind `rc` in place. Returns
/// `false` if the Rc is unexpectedly shared (would indicate a bug in
/// `PdFullSpaceSolver::solve`'s borrow discipline — it should never
/// `Rc::clone` from the rhs vector) or if the block is not a
/// `DenseVector`.
fn write_rhs_block(rc: &mut Rc<dyn pounce_linalg::vector::Vector>, slice: &[Number]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let Some(v) = Rc::get_mut(rc) else {
        return false;
    };
    let Some(dv) = v.as_any_mut().downcast_mut::<DenseVector>() else {
        return false;
    };
    dv.set_values(slice);
    true
}

/// Read the `DenseVector` behind `blk` into `dst`. Handles the
/// homogeneous case (empty z/v blocks for a TNLP with no bounds) by
/// broadcasting the scalar rather than calling `expanded_values()`,
/// which would allocate a fresh `Vec<Number>` every call.
fn read_res_block(blk: &dyn pounce_linalg::vector::Vector, dst: &mut [Number]) -> bool {
    if dst.is_empty() {
        return true;
    }
    let Some(dv) = blk.as_any().downcast_ref::<DenseVector>() else {
        return false;
    };
    if dv.is_homogeneous() {
        let s = dv.scalar();
        for x in dst.iter_mut() {
            *x = s;
        }
    } else {
        dst.copy_from_slice(dv.values());
    }
    true
}

impl PdSensBacksolver {
    /// Single-RHS back-solve against the held factor in the solver's
    /// internal **scaled** space (no natural-units conjugation). This
    /// is the value [`SensBacksolver::solve`] returned before
    /// pounce#128; kept for callers that want the raw factor.
    pub fn solve_scaled_space(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        let total = self.dim();
        if rhs.len() != total || lhs.len() != total {
            return false;
        }
        // Pack rhs into block form.
        let rhs_mut = match self.pack(rhs) {
            Ok(v) => v,
            Err(()) => return false,
        };
        let rhs_iv = rhs_mut.freeze();
        // Fresh result slot, zeroed.
        let mut res_iv = self.template.make_new_zeroed();

        // K · lhs = rhs   ⇒   solve(α=1, β=0, rhs, res) writes
        // res = K⁻¹ · rhs.
        //
        // `allow_inexact = false`: run `PdFullSpaceSolver`'s
        // iterative-refinement loop (`min_refinement_steps = 1`,
        // `residual_ratio_max = 1e-10`) on the held-factor back-solve
        // too, rather than accepting the first substitution the way
        // upstream sIPOPT's `SensSimpleBacksolver` does.
        //
        // It used to be `true`, on the argument that refinement is
        // there to clean up noise during *forward* IPM steps and that
        // the residual it removes here is below `tol` (pounce#77
        // follow-up, where the per-call cost showed up under
        // `jax.jacrev` over a batched `JaxProblem` solve). Two things
        // are wrong with that argument.
        //
        // The first is that "below `tol`" is not the property the
        // callers need. A sens back-solve has no outer loop to
        // self-correct — it is one shot, and its answer is read as a
        // *derivative*, not as a step. `pyomo-pounce`'s covariance /
        // information machinery then asks rank questions about blocks
        // of `K⁻¹`, and `np.linalg.matrix_rank` thresholds the
        // correlation-scaled block at `n · eps` — around `1e-15` for a
        // 2x2. Two coordinates that are exactly dependent (a
        // duplicated design point) produce two rows that agree to
        // whatever accuracy the back-solve has: at `1e-16` they read
        // as dependent and the caller gets its refusal, at `1e-12`
        // they read as independent and it silently gets a covariance
        // for a block that has none. Refinement is what buys those
        // digits.
        //
        // The second is that the premise had a hidden dependency. The
        // unrefined substitution was accurate enough only because the
        // *backend* was refining underneath (`feral_refine` defaulted
        // on); with MA57 — whose `icntl[9] = 0` disables its own
        // refinement — the hole was already open, and turning the
        // feral default off opened it for everyone.
        //
        // The cost the original comment was avoiding is not there to
        // avoid: measured on `kkt_solve_many` with 32 RHS against
        // held factors of dimension 120k/150k/300k (poisson,
        // optcontrol, sparseqp), refined vs unrefined is 0.136/0.120/
        // 0.217 s against 0.138/0.123/0.215 s — inside the noise. The
        // extra substitution runs against the cached matrix, so it
        // never refactorizes, and the pack/unpack around it dominates.
        let allow_inexact = !self.may_refine();
        let ok = {
            let mut pd_ref = self.pd.borrow_mut();
            pd_ref.solve_with_sigma(
                &self.data,
                &self.cq,
                &self.nlp,
                1.0,
                0.0,
                &rhs_iv,
                &mut res_iv,
                allow_inexact,
                /* improve_solution = */ false,
                // Identity on every ordinary solve; the declared-frame
                // diagonal when the held iterate came from crossover
                // (gh#654).
                self.sigma_override(),
            )
        };
        if !ok {
            return false;
        }
        self.unpack(&res_iv, lhs).is_ok()
    }
}

impl SensBacksolver for PdSensBacksolver {
    fn dim(&self) -> usize {
        self.dims.iter().sum()
    }

    /// `F` itself, the vector [`Self::solve`] post-multiplies its
    /// result by, so a caller converting an iterate quantity uses the
    /// same numbers the back-solve used.
    fn natural_units_factor(&self) -> Option<&[Number]> {
        self.conj.as_ref().map(|c| c.f.as_slice())
    }

    fn bound_rows(&self) -> Option<&[crate::backsolver::BoundRow]> {
        self.bound_vars.as_deref().map(|v| v.as_slice())
    }

    fn supports_release(&self) -> bool {
        self.bound_vars.is_some()
    }

    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, false)
    }

    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, true)
    }

    /// Solve `K · lhs = rhs` against the converged factor, in
    /// **natural (unscaled) units** (pounce#128): when the NLP carries
    /// active scaling (`nlp_scaling_method`, `obj_scaling_factor`,
    /// user scaling) the RHS is pre-multiplied by `E` and the result
    /// post-multiplied by `F` (see the `conj` field doc), so
    /// `lhs = K_natural⁻¹ rhs` for **all eight blocks** — including
    /// the z/v bound-multiplier rows. Use
    /// [`Self::solve_scaled_space`] for the raw factor.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        match &self.conj {
            None => self.solve_scaled_space(rhs, lhs),
            Some(c) => {
                let total = self.dim();
                if rhs.len() != total || lhs.len() != total {
                    return false;
                }
                let rhs_scaled: Vec<Number> =
                    rhs.iter().zip(c.e.iter()).map(|(&r, &ei)| r * ei).collect();
                if !self.solve_scaled_space(&rhs_scaled, lhs) {
                    return false;
                }
                for (l, &fi) in lhs.iter_mut().zip(c.f.iter()) {
                    *l *= fi;
                }
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    fn dense(vals: &[Number]) -> Rc<dyn pounce_linalg::Vector> {
        let mut v = DenseVector::new(DenseVectorSpace::new(vals.len() as Index));
        v.values_mut().copy_from_slice(vals);
        Rc::new(v) as Rc<dyn pounce_linalg::Vector>
    }

    /// The ceiling is where the surviving Schur contribution `a²/Σ`
    /// reaches [`SIGMA_PIN_HEADROOM`] ulps, which is what the whole
    /// rule says it is.
    #[test]
    fn the_ceiling_leaves_the_schur_contribution_representable() {
        for a in [1.0, 3.5, 1e-3, 1e6] {
            let ratio = a * a / sigma_pin_cap(a);
            assert!(
                (ratio / (Number::EPSILON * SIGMA_PIN_HEADROOM) - 1.0).abs() < 1e-12,
                "a={a:e}: a²/cap = {ratio:e}, want {:e}",
                Number::EPSILON * SIGMA_PIN_HEADROOM,
            );
        }
    }

    /// A change of variables `x → c·x` sends `Σ → Σ/c²` and `a → a/c`,
    /// so a diagonal that was over its ceiling has to still be over it
    /// afterwards, and one under it has to stay under. Anything keyed
    /// on `a` linearly instead of quadratically fails this.
    ///
    /// The range stops where the ceiling itself saturates — past
    /// `c = 1e3` here the rescaled coefficient is small enough that
    /// [`sigma_pin_cap`] reports no ceiling at all, which is the
    /// deliberate refusal documented there rather than a break in the
    /// invariance.
    #[test]
    fn the_ceiling_is_invariant_to_rescaling_the_variable() {
        for c in [1e-3, 1e-1, 1.0, 1e1, 1e3] {
            let over = 1e27 / (c * c) > sigma_pin_cap(1.0 / c);
            assert!(over, "c={c:e}: rescaling moved the entry under its ceiling");
            let under = 1e6 / (c * c) > sigma_pin_cap(1.0 / c);
            assert!(
                !under,
                "c={c:e}: rescaling moved the entry over its ceiling"
            );
        }
    }

    /// Neither end of the coefficient range may produce a ceiling that
    /// caps something it should not. Huge: no representable `Σ` reaches
    /// the ceiling, so it is no ceiling. Tiny: the ceiling underflows,
    /// and a `Σ` capped at or below `1` is a released bound rather than
    /// a looser pin, so that is no ceiling either.
    #[test]
    fn a_ceiling_that_cannot_help_is_no_ceiling() {
        for a in [Number::MAX, 1e200, 1e-9, 1e-200, Number::MIN_POSITIVE] {
            let cap = sigma_pin_cap(a);
            assert!(
                cap.is_infinite() || cap > 1.0,
                "a={a:e} produced a ceiling of {cap:e}, which would release the bound",
            );
        }
    }

    /// Nothing over its ceiling returns `None`, so an ordinary solve
    /// keeps factoring against the object the calculated quantities
    /// cache rather than an equal copy with a fresh tag.
    #[test]
    fn a_newly_pinned_bound_lands_under_the_ceiling_too() {
        // An addend over the ceiling is held at it, wherever it came
        // from.
        let cap = sigma_pin_cap(1.0);
        let add = 1e-9 / (1e-14 * 1e-14);
        assert!(add > cap, "the fixture needs an addend over the ceiling");
        assert_eq!(pinned_entry(1e6, add, cap), cap);
    }

    /// Where the corrector's addend stands against the ceiling once
    /// `correct_step`'s own clamp is accounted for -- the reason this
    /// path is narrow, in the two numbers that make it narrow. Both
    /// live elsewhere (`corrector.rs` sets the margin, `pinned_rows`
    /// the form of the addend), so this reads as documentation until
    /// one of them moves, which is the point.
    #[test]
    fn the_correctors_clamp_is_what_keeps_its_addend_under_a_unit_ceiling() {
        let mu = 1e-9;
        // `correct_step`: margin = 1e-10 * (1 + |base|), base ~ 1.
        let margin = 1e-10 * 2.0;
        let most = mu / (margin * margin);

        // At a unit coefficient the clamp already does it, three
        // orders clear, and `pinned_entry` is a pass-through.
        let unit = sigma_pin_cap(1.0);
        assert!(most < unit, "{most:e} should sit under {unit:e}");
        assert_eq!(pinned_entry(0.0, most, unit), most);

        // The ceiling binds only once the coefficient brings it down
        // to meet the addend. #738 measured the corrector moving at
        // 1e-3 and 1e-4, and not at 1.
        assert!(sigma_pin_cap(1e-3) < most);
        assert_eq!(
            pinned_entry(0.0, most, sigma_pin_cap(1e-3)),
            sigma_pin_cap(1e-3)
        );

        // The crossover between the two, which is what would move if
        // either the margin or the headroom were retuned.
        let crossover = (mu * Number::EPSILON * SIGMA_PIN_HEADROOM).sqrt() / margin;
        assert!(
            (1e-2..1e-1).contains(&crossover),
            "the corrector's door is this wide: {crossover:e}",
        );
    }

    #[test]
    fn two_representable_contributions_can_swamp_a_row_together() {
        // Neither half is over the ceiling; their sum is. Capping the
        // addend alone would let this through, which is why the
        // ceiling is applied to the entry.
        let cap = sigma_pin_cap(1.0);
        let (had, add) = (0.7 * cap, 0.7 * cap);
        assert!(had < cap && add < cap && had + add > cap);
        assert_eq!(pinned_entry(had, add, cap), cap);
    }

    #[test]
    fn a_pinned_entry_under_the_ceiling_is_just_the_sum() {
        let cap = sigma_pin_cap(1.0);
        assert_eq!(pinned_entry(2.0, 3.0, cap), 5.0);
        // A variable in no constraint row has no ceiling, so the
        // corrector's pin reaches the diagonal whole.
        assert_eq!(pinned_entry(2.0, 1e30, Number::INFINITY), 1e30 + 2.0);
    }

    #[test]
    fn a_diagonal_under_its_ceiling_is_left_alone() {
        let sigma = dense(&[1.0, 1e6, 0.0, 1e12]);
        assert!(cap_sigma(&sigma, &|_| sigma_pin_cap(1.0)).is_none());
    }

    /// Over the ceiling, only the offending entries move.
    #[test]
    fn only_the_entries_over_their_ceiling_move() {
        let cap = sigma_pin_cap(1.0);
        let sigma = dense(&[1.0, 1e27, 1e6, Number::INFINITY]);
        let capped = cap_sigma(&sigma, &|i| if i == 2 { Number::INFINITY } else { cap })
            .expect("two entries are over their ceiling");
        let got = capped
            .as_any()
            .downcast_ref::<DenseVector>()
            .expect("dense")
            .expanded_values();
        assert_eq!(got, vec![1.0, cap, 1e6, cap]);
    }
}
