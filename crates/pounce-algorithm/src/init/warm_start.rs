//! Warm-start iterate initializer — port of
//! `IpWarmStartIterateInitializer.{hpp,cpp}`. Used when a previous
//! solve has left a trial point that should be reused.
//!
//! There are two callers we serve:
//!
//! * **A full primal-dual warm restart** installed via
//!   `Application::set_warm_start_iterate` and consumed by the next
//!   `optimize_tnlp` (e.g. the debugger `resolve` re-solve): `data.curr`
//!   already carries the previous solve's iterate, so we keep it, clamp
//!   multipliers, and optionally override `mu`.
//! * **First solves from `OptimizeTNLP`** that opt into
//!   `warm_start_init_point=yes` to forward user-supplied
//!   primal/dual seeds via `TNLP::get_starting_point`. Here
//!   `data.curr` carries only dim metadata (uninitialized vectors);
//!   we pull seeds from the NLP, push primals/slacks into the bound
//!   interior with warm-start `bound_push`/`bound_frac`, and then
//!   apply the same multiplier clamps.
//!
//! Wired options today: `bound_push`, `bound_frac`,
//! `slack_bound_push`, `slack_bound_frac`, `mult_bound_push`,
//! `mult_init_max`, `target_mu`. `mult_bound_push` floors the four
//! bound-multiplier blocks (mirroring upstream's `ElementWiseMax`
//! with `warm_start_mult_bound_push`): a user-seeded `z = 0` would
//! otherwise start the barrier on its boundary.
//!
//! # Residual-adaptive recentering (gh#606)
//!
//! `warm_start_recentering=residual` (the default) adds a pass over
//! the *supplied* point before the clamps: measure what was actually
//! handed in, reconstruct what is missing, and choose μ from the
//! measurement rather than from a universal constant.
//!
//! 1. **Measure.** `inf_pr` comes first because it is the one residual
//!    that does not depend on the duals, so it is meaningful even when
//!    every multiplier block is absent. It is reported rather than
//!    acted on (step 4 explains why).
//! 2. **Reconstruct bound multipliers.** An entry that arrives as
//!    exactly `0` (or NaN) is not a legal barrier multiplier — the
//!    barrier needs `z > 0` — so it was never a seed. Upstream floors
//!    it at the constant `warm_start_mult_bound_push`; here it takes
//!    `μ̂ / slack` instead, the same complementarity relation the
//!    solver is about to enforce. Seeded (strictly positive) entries
//!    are left alone.
//! 3. **Reconstruct equality multipliers.** A `y` block that is
//!    identically zero is likewise unseeded, and is re-derived from
//!    stationarity by the same regularized least-squares augmented
//!    solve the cold path uses ([`LeastSquareMults`]) — now with the
//!    reconstructed `z` in its right-hand side, so the estimate is not
//!    forced to absorb the bound multipliers.
//! 4. **Choose μ.** With the point complete, μ is raised to the
//!    measured `avrg_compl` when that overshoots what `mu_init` asked
//!    for by more than [`MU_ESCALATION_TRIGGER`], clamped to
//!    `[MU_FLOOR, MU_CEILING]`. A KKT-quality point
//!    measures its own converged complementarity and keeps it; a stale
//!    one, whose multipliers and slacks no longer pair up, measures a
//!    large one and gets a correspondingly loose barrier — the "safely
//!    fall back to stronger recentering" half. The primal and dual
//!    residuals are deliberately *not* in that max; see [`final_mu`].
//!    `warm_start_target_mu` still wins outright when set.
//!
//! `warm_start_recentering=none` restores the pre-gh#606 behaviour
//! exactly: constant floor, zero-filled `y`, μ untouched. It is the
//! kill switch for this whole block.
//!
//! Every branch above records what it did on
//! [`WarmStartDiagnostics`], which lands on `IpoptData` and is
//! readable afterwards through
//! `IpoptApplication::warm_start_diagnostics()`.
//!
//! [`LeastSquareMults`]: crate::eq_mult::least_square::LeastSquareMults

use crate::alg_builder::{WarmStartOptions, WarmStartRecentering};
use crate::eq_mult::least_square::LeastSquareMults;
use crate::eq_mult::r#trait::EqMultCalculator;
use crate::init::default::push_x_into_interior;
use crate::init::r#trait::IterateInitializer;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::aug_system_solver::AugSystemSolver;
use pounce_common::types::Number;
use pounce_linalg::Vector;
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use std::cell::RefCell;
use std::rc::Rc;

/// Hard floor on the residual-derived μ. Below this the barrier term
/// is at the noise level of a `tol=1e-8` solve's complementarity and
/// carries no information; letting a measurement drive μ further down
/// would start the solve on a boundary the line search then has to
/// climb back off.
const MU_FLOOR: Number = 1e-11;

/// Hard ceiling on the residual-derived μ: upstream's registered
/// `mu_init` default. A warm start that measures badly should degrade
/// *to* a cold start, never past one.
const MU_CEILING: Number = 0.1;

/// How far the measured complementarity has to exceed `mu_init` before
/// it is allowed to override it.
///
/// Moving μ reroutes the entire trajectory, so the move has to be
/// worth it. The cases this exists for — a stale seed, a caller who
/// kept only the primal point — miss by three to six orders of
/// magnitude; a good seed misses by a factor of two, and overriding
/// there buys nothing and costs whatever the reroute costs. On
/// `cresc4` (a nonconvex model that enters restoration on iteration 2,
/// where small perturbations compound) a measurement of `5e-7` against
/// a `mu_init` of `1e-7` moved μ by half an order and the solve from
/// 85 iterations to 206 — same status, same objective, 2.4x the work.
/// With this gate that model is bit-identical again, and the
/// three-order cases still fire.
const MU_ESCALATION_TRIGGER: Number = 10.0;

/// What happened to one multiplier block of the supplied warm point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockVerdict {
    /// The block has dimension zero — the model has no such block.
    #[default]
    Absent,
    /// Every entry arrived seeded and was kept (modulo the clamps).
    Accepted,
    /// Some or all entries arrived unseeded and were rebuilt.
    Reconstructed,
    /// A reconstruction ran and was thrown away (it exceeded
    /// `constr_mult_init_max`, or the augmented solve failed); the
    /// block fell back to the pre-gh#606 constant fill.
    Discarded,
    /// The caller supplied no dual information *at all*, so there was
    /// nothing to reconstruct this block from and it kept the
    /// pre-gh#606 constant fill. See [`any_dual_seeded`].
    Unseeded,
}

/// What the warm-start initializer accepted, reconstructed, or
/// discarded, and the residuals it based those calls on (gh#606).
///
/// Written once per solve, at initialization. `mu_in` is what
/// `mu_init` / `warm_start_target_mu` asked for; `mu_out` is what the
/// iterate actually starts at.
#[derive(Debug, Clone)]
pub struct WarmStartDiagnostics {
    /// `‖c(x)‖_∞` of the supplied primal point, measured before any
    /// dual reconstruction (it does not depend on the duals).
    pub primal_residual: Number,
    /// `‖∇_x L‖_∞` after reconstruction.
    pub dual_residual: Number,
    /// Average complementarity after reconstruction.
    pub complementarity: Number,
    pub mu_in: Number,
    pub mu_out: Number,
    pub bound_duals: BlockVerdict,
    pub eq_duals: BlockVerdict,
    /// Bound-multiplier entries that arrived unseeded and were filled
    /// from `μ̂ / slack` rather than from the constant floor.
    pub bound_duals_reconstructed: usize,
    /// `true` when the unseeded bound multipliers were re-derived from
    /// the stationarity identity rather than left at `μ̂ / slack`. That
    /// needs a caller-supplied `y`; see
    /// [`refine_bound_duals_from_stationarity`].
    pub stationarity_split: bool,
    /// `true` when `warm_start_recentering=none` turned all of the
    /// above off and the fields are the legacy constants.
    pub recentering_disabled: bool,
}

impl Default for WarmStartDiagnostics {
    fn default() -> Self {
        Self {
            primal_residual: Number::NAN,
            dual_residual: Number::NAN,
            complementarity: Number::NAN,
            mu_in: Number::NAN,
            mu_out: Number::NAN,
            bound_duals: BlockVerdict::Absent,
            eq_duals: BlockVerdict::Absent,
            bound_duals_reconstructed: 0,
            stationarity_split: false,
            recentering_disabled: false,
        }
    }
}

pub struct WarmStartIterateInitializer {
    opts: WarmStartOptions,
}

impl WarmStartIterateInitializer {
    pub fn new() -> Self {
        Self {
            opts: WarmStartOptions::default(),
        }
    }

    pub fn with_options(opts: WarmStartOptions) -> Self {
        Self { opts }
    }
}

impl Default for WarmStartIterateInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl IterateInitializer for WarmStartIterateInitializer {
    fn set_initial_iterates(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
    ) -> bool {
        // Two entry points share this initializer: the re-optimize path
        // (curr.x carries values from the prior solve) and the first
        // OptimizeTNLP call that opted into warm_start_init_point=yes
        // (curr.x is the application's placeholder seed — allocated but
        // never written). Detect the latter and rebuild `curr` from the
        // NLP's get_starting_x/y/z hooks before clamping.
        let needs_seed_from_nlp = {
            let borrow = data.borrow();
            match borrow.curr.as_ref() {
                None => return false,
                Some(c) => !is_initialized(&c.x),
            }
        };

        if needs_seed_from_nlp && !seed_from_nlp(data, nlp, &self.opts) {
            return false;
        }

        let mut diag = WarmStartDiagnostics {
            mu_in: data.borrow().curr_mu,
            recentering_disabled: self.opts.recentering == WarmStartRecentering::None,
            ..Default::default()
        };

        // gh#606 steps 1-3: measure the supplied point, then rebuild
        // whatever it did not carry. Runs *before* the clamp block
        // below, which still has the last word on the caps.
        //
        // Reconstruction completes a *partial* warm start: each missing
        // block is derived from the blocks that were supplied. When
        // nothing at all was supplied there is nothing to derive from,
        // and what comes out is the cold path's estimate wearing the
        // warm path's barrier — measured over `benchmarks/warmstart`,
        // reconstructing from a primal-only seed cost 1102 -> 1211
        // iterations across 27 parametric paths (`degenerate_corner`
        // 17 -> 38 at every step size). So a seed with no duals in it
        // keeps the pre-gh#606 fills, and only μ is still measured.
        let dual_info =
            self.opts.recentering == WarmStartRecentering::Residual && any_dual_seeded(data);
        let mu_hat = if dual_info {
            let (mu_hat, unseeded) = recenter_from_residuals(data, cq, &self.opts, &mut diag);
            reconstruct_eq_duals(data, cq, nlp, aug_solver, &self.opts, &mut diag);
            // The split below reads `y`; running it against a `y` this
            // same pass just derived *from* the provisional `z` is
            // circular, and measurably so (see the fn docs). Only a
            // caller-supplied `y` earns it.
            if diag.eq_duals != BlockVerdict::Reconstructed {
                refine_bound_duals_from_stationarity(data, cq, nlp, &self.opts, mu_hat, &unseeded);
                diag.stationarity_split = true;
            }
            Some(mu_hat)
        } else if self.opts.recentering == WarmStartRecentering::Residual {
            diag.primal_residual = cq.borrow().curr_primal_infeasibility_max();
            diag.bound_duals = BlockVerdict::Unseeded;
            diag.eq_duals = BlockVerdict::Unseeded;
            Some(safe_mu(data.borrow().curr_mu))
        } else {
            None
        };

        {
            // Rebuild `curr` with clamped multipliers. Components are
            // shared via `Rc` with previous solves, so we make fresh
            // copies before mutating to avoid clobbering downstream
            // borrowers. Bound multipliers are additionally floored at
            // `mult_bound_push` (upstream `warm_start_mult_bound_push`):
            // the barrier needs them strictly positive, and a carried-in
            // 0 (e.g. an inactive bound in the previous solution) would
            // otherwise start on the boundary. This block runs even
            // with both clamps disabled (cap = inf, floor = 0; the
            // floor still clamps a negative z/v to 0) because it also
            // resolves NaN seeds: NaN in a user-supplied multiplier
            // means "unseeded", and takes `bound_mult_init_val` for
            // bound multipliers, or 0 for equality multipliers. That 0
            // is the warm path's existing unseeded value (what
            // `seed_from_nlp` produced already).
            //
            // Under `warm_start_recentering=residual` the unseeded
            // entries have already been rebuilt above, so what reaches
            // here is a complete point and the constants only act as
            // the outer caps.
            let mut borrow = data.borrow_mut();
            let curr = borrow.curr.as_ref().unwrap();
            let cap = if self.opts.mult_init_max > 0.0 {
                self.opts.mult_init_max
            } else {
                f64::INFINITY
            };
            let z_floor = self.opts.mult_bound_push.max(0.0);
            let z_nan = self.opts.bound_mult_init_val;
            let new_curr = IteratesVector::new(
                Rc::clone(&curr.x),
                Rc::clone(&curr.s),
                clone_clamped(&curr.y_c, -cap, cap, 0.0),
                clone_clamped(&curr.y_d, -cap, cap, 0.0),
                clone_clamped(&curr.z_l, z_floor, cap, z_nan),
                clone_clamped(&curr.z_u, z_floor, cap, z_nan),
                clone_clamped(&curr.v_l, z_floor, cap, z_nan),
                clone_clamped(&curr.v_u, z_floor, cap, z_nan),
            );
            borrow.set_curr(new_curr);
        }

        // `warm_start_target_mu` is an explicit instruction and still
        // wins outright; the residual estimate only fills the gap the
        // user left. gh#606 step 4.
        if self.opts.target_mu > 0.0 {
            data.borrow_mut().curr_mu = self.opts.target_mu;
        } else if mu_hat.is_some() {
            let mu = final_mu(data, cq, &mut diag);
            data.borrow_mut().curr_mu = mu;
        }

        {
            let mut borrow = data.borrow_mut();
            diag.mu_out = borrow.curr_mu;
            for token in diag.info_string_tokens() {
                borrow.append_info_string(token);
            }
            borrow.warm_start_diagnostics = Some(diag);
        }

        true
    }
}

impl WarmStartDiagnostics {
    /// Single-token summaries for the iteration line, in the same
    /// spirit as the cold path's `y` / `y0` / `yc`
    /// (`DefaultIterateInitializer`). `wz` = bound multipliers
    /// reconstructed, `wy` = equality multipliers reconstructed,
    /// `wy0` = a reconstruction was discarded, `wmu` = μ was raised
    /// above what `mu_init` asked for (the stale-point fallback).
    fn info_string_tokens(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.bound_duals == BlockVerdict::Reconstructed {
            out.push("wz");
        }
        match self.eq_duals {
            BlockVerdict::Reconstructed => out.push("wy"),
            BlockVerdict::Discarded => out.push("wy0"),
            _ => {}
        }
        if self.mu_out > self.mu_in * 10.0 {
            out.push("wmu");
        }
        out
    }
}

/// gh#606 steps 1-2. Measure the dual-free primal residual of the
/// supplied point, turn it into a provisional barrier parameter `μ̂`,
/// and fill every *unseeded* bound-multiplier entry with `μ̂ / slack`.
///
/// "Unseeded" is `0` exactly, or NaN. Neither is a legal barrier
/// multiplier — the barrier needs `z > 0` — so neither can have come
/// from a converged solve, and both are already special-cased today
/// (floored at the constant `warm_start_mult_bound_push`). What
/// changes is the value they take.
///
/// Returns `μ̂` — which [`final_mu`] then refines once the whole point
/// is assembled — and, per block, the mask of entries that were
/// unseeded, so [`refine_bound_duals_from_stationarity`] can revisit
/// exactly those and nothing else.
fn recenter_from_residuals(
    data: &IpoptDataHandle,
    cq: &IpoptCqHandle,
    opts: &WarmStartOptions,
    diag: &mut WarmStartDiagnostics,
) -> (Number, [Vec<bool>; 4]) {
    let inf_pr = cq.borrow().curr_primal_infeasibility_max();
    diag.primal_residual = inf_pr;

    // The fill barrier is the one the *caller* asked for, not the one
    // the residual will argue for below. `z_i = μ / slack_i` is the
    // complementarity relation at barrier μ, so filling from `mu_init`
    // makes the reconstructed multipliers consistent with the point
    // that produced the slacks — on an exact restart it reproduces the
    // converged multiplier to a few digits. Substituting a
    // residual-inflated μ here instead multiplies every reconstructed
    // entry by that inflation, which on a converged point (tiny slacks)
    // is precisely where it does the most damage: measured on HS071, a
    // 8x-inflated μ̂ put the reconstructed slack multiplier 8x high and
    // took the exact restart from 1 iteration to 5. The residual has
    // its say in [`final_mu`], after the point is assembled.
    let mu_hat = safe_mu(data.borrow().curr_mu);

    let mut unseeded: [Vec<bool>; 4] = [vec![], vec![], vec![], vec![]];
    let curr = match data.borrow().curr.clone() {
        Some(c) => c,
        None => return (mu_hat, unseeded),
    };
    let cq_ref = cq.borrow();
    let slacks = [
        cq_ref.curr_slack_x_l(),
        cq_ref.curr_slack_x_u(),
        cq_ref.curr_slack_s_l(),
        cq_ref.curr_slack_s_u(),
    ];
    drop(cq_ref);
    let blocks = [&curr.z_l, &curr.z_u, &curr.v_l, &curr.v_u];

    let mut rebuilt: [Option<Rc<dyn Vector>>; 4] = [None, None, None, None];
    let mut n_reconstructed = 0usize;
    let mut n_total = 0usize;
    for (i, (z, slack)) in blocks.iter().zip(slacks.iter()).enumerate() {
        if z.dim() == 0 {
            continue;
        }
        n_total += z.dim() as usize;
        let Some(mut vals) = flatten(&***z) else {
            continue;
        };
        let Some(sl) = flatten(&**slack) else {
            continue;
        };
        if sl.len() != vals.len() {
            // Shapes disagree (a layout this helper does not model):
            // leave the block to the constant floor below rather than
            // pair up entries that may not correspond.
            continue;
        }
        let mut touched = 0usize;
        let mut mask = vec![false; vals.len()];
        for (k, (v, s)) in vals.iter_mut().zip(sl.iter()).enumerate() {
            if !(v.is_nan() || *v == 0.0) {
                continue;
            }
            touched += 1;
            mask[k] = true;
            // `slack` can be tiny (an active bound) or non-finite on a
            // point outside its bounds; both are clamped into the band
            // the caps below would have allowed anyway.
            let filled = if s.is_finite() && *s > 0.0 {
                mu_hat / *s
            } else {
                mu_hat
            };
            *v = filled.clamp(
                opts.mult_bound_push.max(MU_FLOOR),
                opts.mult_init_max_or_inf(),
            );
        }
        if touched == 0 {
            continue;
        }
        unseeded[i] = mask;
        n_reconstructed += touched;
        let mut out = z.make_new();
        if scatter(&mut *out, &vals) {
            rebuilt[i] = Some(Rc::from(out));
        }
    }

    if n_total > 0 {
        diag.bound_duals = if n_reconstructed == 0 {
            BlockVerdict::Accepted
        } else {
            BlockVerdict::Reconstructed
        };
    }
    diag.bound_duals_reconstructed = n_reconstructed;

    if rebuilt.iter().any(|r| r.is_some()) {
        let pick = |i: usize, orig: &Rc<dyn Vector>| -> Rc<dyn Vector> {
            rebuilt[i].clone().unwrap_or_else(|| Rc::clone(orig))
        };
        let new_curr = IteratesVector::new(
            Rc::clone(&curr.x),
            Rc::clone(&curr.s),
            Rc::clone(&curr.y_c),
            Rc::clone(&curr.y_d),
            pick(0, &curr.z_l),
            pick(1, &curr.z_u),
            pick(2, &curr.v_l),
            pick(3, &curr.v_u),
        );
        data.borrow_mut().set_curr(new_curr);
    }

    (mu_hat, unseeded)
}

/// gh#606 step 3b. Having reconstructed the equality multipliers,
/// revisit the bound multipliers that were unseeded and replace the
/// `μ̂ / slack` guess with the value stationarity actually implies.
///
/// `∇_x L = ∇f + J_cᵀ y_c + J_dᵀ y_d − P_L z_L + P_U z_U`, so at a
/// stationary point `P_L z_L − P_U z_U = r_x` with
/// `r_x = ∇f + J_cᵀ y_c + J_dᵀ y_d`; splitting `r_x` by sign and
/// selecting through `P_Lᵀ` / `P_Uᵀ` is the positivity-preserving
/// solution of that identity. The slack block is the same identity one
/// row down: `∇_s L = −y_d − P_L v_L + P_U v_U`, so `−y_d` splits into
/// `v_L` / `v_U`.
///
/// Only runs when the equality multipliers came from the caller. When
/// they were themselves reconstructed a few lines earlier, `r_x` is
/// built from a `y` that the least-squares solve derived *from* the
/// provisional `μ̂ / slack` fill, so the split re-derives its own
/// input — and the round trip is lossy at exactly the points that make
/// it hard. Measured over `benchmarks/warmstart`, running it anyway
/// cost a primal-only warm start 1102 -> 1263 iterations across 27
/// paths (`degenerate_corner` 17 -> 38, `rosenbrock_ring` 28 -> 66),
/// with the wins concentrated where `y` *was* supplied.
///
/// Why this and not `μ̂ / slack` alone: the slacks reaching here have
/// already been shoved into the bound interior by
/// `warm_start_slack_bound_push`, so on a converged point — where the
/// true slack sits at `μ / z` and the push dominates it — `μ̂ / slack`
/// is off by exactly the push's inflation. Measured on HS071, that put
/// the reconstructed inequality multiplier 5.5x low and took an exact
/// restart from 1 iteration to 5. The stationarity split is immune to
/// it because it never looks at a slack.
///
/// The `μ̂ / slack` value survives as the *floor*: at an inactive bound
/// the split contributes nothing and complementarity is what says how
/// big the multiplier should be.
fn refine_bound_duals_from_stationarity(
    data: &IpoptDataHandle,
    cq: &IpoptCqHandle,
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    opts: &WarmStartOptions,
    mu_hat: Number,
    unseeded: &[Vec<bool>; 4],
) {
    if unseeded.iter().all(|m| m.is_empty()) {
        return;
    }
    let curr = match data.borrow().curr.clone() {
        Some(c) => c,
        None => return,
    };

    // r_x = ∇f + J_cᵀ y_c + J_dᵀ y_d, and r_s = −y_d.
    let (r_x, r_s, slacks) = {
        let cq_ref = cq.borrow();
        let grad_f = cq_ref.curr_grad_f();
        let jc_t = cq_ref.curr_jac_c_t_times_curr_y_c();
        let jd_t = cq_ref.curr_jac_d_t_times_curr_y_d();
        let mut r_x = grad_f.make_new();
        r_x.copy(&*grad_f);
        r_x.add_two_vectors(1.0, &*jc_t, 1.0, &*jd_t, 1.0);
        let mut r_s = curr.y_d.make_new();
        r_s.copy(&*curr.y_d);
        r_s.scal(-1.0);
        let slacks = [
            cq_ref.curr_slack_x_l(),
            cq_ref.curr_slack_x_u(),
            cq_ref.curr_slack_s_l(),
            cq_ref.curr_slack_s_u(),
        ];
        (r_x, r_s, slacks)
    };

    let nlp_ref = nlp.borrow();
    // Rearranged, the two identities read `P_L z_L − P_U z_U = r_x`
    // and `P_L v_L − P_U v_U = r_s`, so each lower block takes `+Pᵀr`
    // and each upper block `−Pᵀr`; the `max(·, 0)` that makes the
    // split well-posed is the `.max(floor)` in the loop below.
    let targets: [Option<Vec<Number>>; 4] = [
        project(&*r_x, &*nlp_ref.px_l(), 1.0, &curr.z_l),
        project(&*r_x, &*nlp_ref.px_u(), -1.0, &curr.z_u),
        project(&*r_s, &*nlp_ref.pd_l(), 1.0, &curr.v_l),
        project(&*r_s, &*nlp_ref.pd_u(), -1.0, &curr.v_u),
    ];
    drop(nlp_ref);

    let blocks = [&curr.z_l, &curr.z_u, &curr.v_l, &curr.v_u];
    let mut rebuilt: [Option<Rc<dyn Vector>>; 4] = [None, None, None, None];
    for (i, block) in blocks.iter().enumerate() {
        let mask = &unseeded[i];
        if mask.is_empty() {
            continue;
        }
        let (Some(target), Some(mut vals), Some(sl)) =
            (targets[i].clone(), flatten(&***block), flatten(&*slacks[i]))
        else {
            continue;
        };
        if target.len() != vals.len() || sl.len() != vals.len() || mask.len() != vals.len() {
            continue;
        }
        let cap = opts.mult_init_max_or_inf();
        let hard_floor = opts.mult_bound_push.max(MU_FLOOR);
        for k in 0..vals.len() {
            if !mask[k] {
                continue;
            }
            let compl_floor = if sl[k].is_finite() && sl[k] > 0.0 {
                mu_hat / sl[k]
            } else {
                mu_hat
            };
            let split = if target[k].is_finite() {
                target[k]
            } else {
                0.0
            };
            vals[k] = split.max(compl_floor).max(hard_floor).min(cap);
        }
        let mut out = block.make_new();
        if scatter(&mut *out, &vals) {
            rebuilt[i] = Some(Rc::from(out));
        }
    }

    if rebuilt.iter().all(|r| r.is_none()) {
        return;
    }
    let pick = |i: usize, orig: &Rc<dyn Vector>| -> Rc<dyn Vector> {
        rebuilt[i].clone().unwrap_or_else(|| Rc::clone(orig))
    };
    let new_curr = IteratesVector::new(
        Rc::clone(&curr.x),
        Rc::clone(&curr.s),
        Rc::clone(&curr.y_c),
        Rc::clone(&curr.y_d),
        pick(0, &curr.z_l),
        pick(1, &curr.z_u),
        pick(2, &curr.v_l),
        pick(3, &curr.v_u),
    );
    data.borrow_mut().set_curr(new_curr);
}

/// `sign · Pᵀ r`, as a plain slice of length `n_out`. `P` is one of the
/// packed bound-selection matrices, so the transpose picks out the
/// components that actually carry that bound.
fn project(
    r: &dyn Vector,
    p: &dyn pounce_linalg::Matrix,
    sign: Number,
    template: &Rc<dyn Vector>,
) -> Option<Vec<Number>> {
    if template.dim() == 0 {
        return None;
    }
    let mut out = template.make_new();
    out.set(0.0);
    p.trans_mult_vector(sign, r, 0.0, &mut *out);
    flatten(&*out)
}

/// gh#606 step 3. Re-derive an identically-zero `y` block from
/// stationarity.
///
/// A converged equality multiplier vector is zero only when every
/// equality is inactive in the Lagrangian, which the least-squares
/// solve reproduces anyway — so treating an all-zero block as
/// "unseeded" is self-correcting on a genuinely-zero one, and is the
/// only signal available: an absent `lagrange=` seed and a supplied
/// vector of zeros reach this initializer as the same bytes.
///
/// This is the same [`LeastSquareMults`] augmented solve the cold path
/// runs, so the "sparse regularized stationarity least-squares"
/// machinery is shared rather than duplicated — the difference is that
/// here it runs with real seeded bound multipliers in its right-hand
/// side instead of the cold path's constant `bound_mult_init_val`.
fn reconstruct_eq_duals(
    data: &IpoptDataHandle,
    cq: &IpoptCqHandle,
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    aug_solver: &mut dyn AugSystemSolver,
    opts: &WarmStartOptions,
    diag: &mut WarmStartDiagnostics,
) {
    let curr = match data.borrow().curr.clone() {
        Some(c) => c,
        None => return,
    };
    let (n_yc, n_yd) = (curr.y_c.dim(), curr.y_d.dim());
    if n_yc + n_yd == 0 {
        return;
    }
    // Upstream's own guard from the cold path: with as many equalities
    // as variables the least-squares system is square and the estimate
    // is not a projection of anything.
    if n_yc == curr.x.dim() {
        diag.eq_duals = BlockVerdict::Accepted;
        return;
    }
    let seeded = !is_identically_zero(&curr.y_c) || !is_identically_zero(&curr.y_d);
    if seeded {
        diag.eq_duals = BlockVerdict::Accepted;
        return;
    }

    let mut new_y_c = DenseVectorSpace::new(n_yc).make_new_dense();
    let mut new_y_d = DenseVectorSpace::new(n_yd).make_new_dense();
    new_y_c.set(0.0);
    new_y_d.set(0.0);
    let ok = LeastSquareMults::new().calculate_y_eq(
        data,
        cq,
        nlp,
        aug_solver,
        &mut new_y_c,
        &mut new_y_d,
    );
    if !ok {
        diag.eq_duals = BlockVerdict::Discarded;
        return;
    }
    let norm = new_y_c.amax().max(new_y_d.amax());
    if !norm.is_finite() || (opts.constr_mult_init_max > 0.0 && norm > opts.constr_mult_init_max) {
        // Same verdict, and the same cap, the cold path reaches on an
        // over-large estimate: keep the zeros rather than start the
        // solve on a multiplier the cap was written to exclude. On a
        // rank-deficient model the least-squares system is singular and
        // this is the branch that catches it.
        diag.eq_duals = BlockVerdict::Discarded;
        return;
    }
    diag.eq_duals = BlockVerdict::Reconstructed;
    let new_curr = IteratesVector::new(
        Rc::clone(&curr.x),
        Rc::clone(&curr.s),
        Rc::new(new_y_c),
        Rc::new(new_y_d),
        Rc::clone(&curr.z_l),
        Rc::clone(&curr.z_u),
        Rc::clone(&curr.v_l),
        Rc::clone(&curr.v_u),
    );
    data.borrow_mut().set_curr(new_curr);
}

/// gh#606 step 4. The barrier parameter the assembled point deserves.
///
/// Measured on the reconstructed iterate, so `avrg_compl` reflects the
/// multipliers the solve will actually start from rather than the
/// holes the caller left.
///
/// **Only the complementarity moves μ.** That is not an oversight: of
/// the three KKT residuals, complementarity is the one μ *is*, and the
/// other two are what the Newton step is for. A warm point at a
/// slightly moved parameter carries a primal and a dual residual of
/// order `Δθ` by construction — that is the premise of warm starting —
/// and raising μ to meet them throws away the warm start to pay for a
/// step the solver was about to take anyway. Measured over the
/// `benchmarks/warmstart` corpus, a `μ ≥ κ·max(inf_pr, inf_du)` rule
/// cost 715 → 1129 iterations across 27 parametric paths: on
/// `simplex_proj/tiny` a re-solve that needed one iteration measured
/// `inf_du = 2e-3`, took `μ = 2e-3`, and needed five. `avrg_compl` on
/// the same point read `2.6e-9` — the converged barrier, correctly
/// recognised. A genuinely stale point is still caught, because its
/// multipliers and its slacks no longer pair up and `avrg_compl` rises
/// with them.
///
/// `mu_in` is a floor, never a ceiling: `mu_init` is an explicit
/// statement about the barrier the caller wants, and the measurement
/// is here to catch a point that cannot support it, not to
/// second-guess a good one downward. It also has to miss by
/// [`MU_ESCALATION_TRIGGER`] before it is overridden at all.
fn final_mu(data: &IpoptDataHandle, cq: &IpoptCqHandle, diag: &mut WarmStartDiagnostics) -> Number {
    let (compl, inf_du) = {
        let cq_ref = cq.borrow();
        (
            cq_ref.curr_avrg_compl(),
            cq_ref.curr_dual_infeasibility_max(),
        )
    };
    diag.complementarity = compl;
    // Recorded but deliberately not fed into μ — see above. It is the
    // number that says whether the reconstruction worked.
    diag.dual_residual = inf_du;

    let mu_in = data.borrow().curr_mu;
    let mut mu = mu_in;
    if compl.is_finite() && compl > MU_ESCALATION_TRIGGER * mu_in {
        mu = compl;
    }
    safe_mu(mu).max(mu_in.min(MU_CEILING))
}

/// Clamp a candidate μ into the band a warm start may use, and reject
/// non-finite candidates outright.
fn safe_mu(mu: Number) -> Number {
    if !mu.is_finite() || mu <= 0.0 {
        return MU_CEILING;
    }
    mu.clamp(MU_FLOOR, MU_CEILING)
}

/// Copy a vector's scalars out, whatever storage it uses. `None` for a
/// layout this module does not model, or one that was never written.
fn flatten(v: &dyn Vector) -> Option<Vec<Number>> {
    if let Some(d) = v.as_any().downcast_ref::<DenseVector>() {
        // `expanded_values`, not `values`: a block written with
        // `set(c)` is stored homogeneously and `values` asserts on it.
        return d.is_initialized().then(|| d.expanded_values());
    }
    if let Some(c) = v.as_any().downcast_ref::<CompoundVector>() {
        let mut out = Vec::with_capacity(v.dim() as usize);
        for i in 0..c.n_comps() {
            out.extend(flatten(c.comp(i))?);
        }
        return Some(out);
    }
    None
}

/// Inverse of [`flatten`]. `false` when the layout or the length does
/// not match, in which case the caller must leave the block alone.
fn scatter(v: &mut dyn Vector, src: &[Number]) -> bool {
    if v.dim() as usize != src.len() {
        return false;
    }
    if v.as_any().is::<DenseVector>() {
        let d = v.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        d.values_mut().copy_from_slice(src);
        return true;
    }
    if v.as_any().is::<CompoundVector>() {
        let c = v.as_any_mut().downcast_mut::<CompoundVector>().unwrap();
        let mut off = 0usize;
        for i in 0..c.n_comps() {
            let comp = c.comp_mut(i);
            let n = comp.dim() as usize;
            if !scatter(comp, &src[off..off + n]) {
                return false;
            }
            off += n;
        }
        return true;
    }
    false
}

/// `true` when every entry is exactly `0` — the signature of an
/// equality-multiplier block that no caller seeded. An uninitialized
/// block counts as zero: that is what the clamp step below fills it
/// with.
fn is_identically_zero(v: &Rc<dyn Vector>) -> bool {
    if v.dim() == 0 {
        return false;
    }
    match flatten(&**v) {
        Some(vals) => vals.iter().all(|e| *e == 0.0),
        // Never written -> the clamp block collapses it to zero.
        None => true,
    }
}

/// Pull a fresh starting iterate from the NLP (which routes to
/// `TNLP::get_starting_point` with `init_x` / `init_lambda` /
/// `init_z` all true), push the primals and slacks into the bound
/// interior using warm-start-specific `bound_push`/`bound_frac`, and
/// install the result on `data.curr`. Mirrors steps 1-4 of
/// `DefaultIterateInitializer::set_initial_iterates`, but with
/// upstream's warm-start option block governing the push.
fn seed_from_nlp(
    data: &IpoptDataHandle,
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    opts: &WarmStartOptions,
) -> bool {
    if !nlp.borrow_mut().prepare_warm_start() {
        return false;
    }
    let (n_x, n_s, n_yc, n_yd, n_zl, n_zu, n_vl, n_vu) = {
        let borrow = data.borrow();
        let c = borrow.curr.as_ref().unwrap();
        (
            c.x.dim(),
            c.s.dim(),
            c.y_c.dim(),
            c.y_d.dim(),
            c.z_l.dim(),
            c.z_u.dim(),
            c.v_l.dim(),
            c.v_u.dim(),
        )
    };

    let mut x = DenseVectorSpace::new(n_x).make_new_dense();
    nlp.borrow_mut().get_starting_x(&mut x);
    {
        let nlp_ref = nlp.borrow();
        push_x_into_interior(
            &mut x,
            &*nlp_ref.px_l(),
            nlp_ref.x_l(),
            &*nlp_ref.px_u(),
            nlp_ref.x_u(),
            opts.bound_push,
            opts.bound_frac,
        );
    }

    let mut s = DenseVectorSpace::new(n_s).make_new_dense();
    nlp.borrow_mut().eval_d(&x, &mut s);
    {
        let nlp_ref = nlp.borrow();
        push_x_into_interior(
            &mut s,
            &*nlp_ref.pd_l(),
            nlp_ref.d_l(),
            &*nlp_ref.pd_u(),
            nlp_ref.d_u(),
            opts.slack_bound_push,
            opts.slack_bound_frac,
        );
    }

    let mut y_c = DenseVectorSpace::new(n_yc).make_new_dense();
    let mut y_d = DenseVectorSpace::new(n_yd).make_new_dense();
    y_c.set(0.0);
    y_d.set(0.0);
    nlp.borrow_mut().get_starting_y(&mut y_c, &mut y_d);

    let mut z_l = DenseVectorSpace::new(n_zl).make_new_dense();
    let mut z_u = DenseVectorSpace::new(n_zu).make_new_dense();
    let mut v_l = DenseVectorSpace::new(n_vl).make_new_dense();
    let mut v_u = DenseVectorSpace::new(n_vu).make_new_dense();
    z_l.set(0.0);
    z_u.set(0.0);
    v_l.set(0.0);
    v_u.set(0.0);
    nlp.borrow_mut()
        .get_starting_z(&mut z_l, &mut z_u, &mut v_l, &mut v_u);
    nlp.borrow_mut().finish_warm_start();

    let iv = IteratesVector::new(
        Rc::new(x),
        Rc::new(s),
        Rc::new(y_c),
        Rc::new(y_d),
        Rc::new(z_l),
        Rc::new(z_u),
        Rc::new(v_l),
        Rc::new(v_u),
    );
    data.borrow_mut().set_curr(iv);
    true
}

fn is_initialized(v: &Rc<dyn Vector>) -> bool {
    if v.dim() == 0 {
        return true;
    }
    v.as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.is_initialized())
        .unwrap_or(true)
}

/// Replace every NaN entry of `v` with `fill`, in place.
///
/// NaN in a user-supplied multiplier seed means "unseeded" (see the
/// `Problem.solve` contract), and has to be resolved before the
/// clamps: `element_wise_min`/`element_wise_max` would propagate it
/// into the iterate, poisoning the solve.
///
/// Both `Vector` storage layouts are handled. A dense block is
/// scanned directly; a compound block recurses into its components,
/// so the contract holds wherever the iterate's multiplier blocks
/// live — the seed path (`seed_from_nlp`) always builds dense
/// vectors, but the re-optimize path reuses whatever the previous
/// solve's spaces produced, and a debug-only guard would be compiled
/// out of exactly the release builds that ship.
fn resolve_nan_seeds(v: &mut dyn Vector, fill: f64) {
    // Type-test before taking the mutable borrow: `if let Some(d) =
    // v.as_any_mut()… else` would keep that borrow live across the
    // else arm.
    if v.as_any().is::<DenseVector>() {
        let d = v.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        for e in d.values_mut() {
            if e.is_nan() {
                *e = fill;
            }
        }
    } else if v.as_any().is::<CompoundVector>() {
        let c = v.as_any_mut().downcast_mut::<CompoundVector>().unwrap();
        for i in 0..c.n_comps() {
            resolve_nan_seeds(c.comp_mut(i), fill);
        }
    } else {
        // `DenseVector` and `CompoundVector` are the only `Vector`
        // implementations; a third one must be handled here, or NaN
        // rides the clamps into the iterate as a silent poison.
        debug_assert!(false, "resolve_nan_seeds: unhandled Vector implementation");
    }
}

/// Clone `v` into a fresh owned vector and clamp every entry to
/// `[lo, hi]` componentwise. Empty vectors short-circuit. Vectors that
/// were never written to (the application's placeholder seed iterates
/// before any solve ran) collapse to a zero-initialized vector — `0`
/// is inside every well-formed warm-start clamp range, so this matches
/// upstream's behavior when a multiplier block has no carry-over
/// value.
fn clone_clamped(v: &Rc<dyn Vector>, lo: f64, hi: f64, nan_fill: f64) -> Rc<dyn Vector> {
    let n = v.dim();
    if n == 0 {
        return Rc::clone(v);
    }
    let mut out = v.make_new();
    let initialized = v
        .as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.is_initialized())
        .unwrap_or(true);
    if initialized {
        out.copy(&**v);
        // NaN marks an unseeded entry; resolve it before the clamps
        // (element-wise min/max would just propagate it)
        resolve_nan_seeds(&mut *out, nan_fill);
    } else {
        out.set(0.0);
    }
    let mut cap_hi = v.make_new();
    cap_hi.set(hi);
    out.element_wise_min(&*cap_hi);
    let mut cap_lo = v.make_new();
    cap_lo.set(lo);
    out.element_wise_max(&*cap_lo);
    Rc::from(out)
}

#[cfg(test)]
mod tests_nan_seed {
    use super::*;
    use pounce_linalg::compound_vector::CompoundVectorSpace;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    #[test]
    fn nan_entries_take_the_fill_before_clamping() {
        let space = DenseVectorSpace::new(3);
        let mut d = space.make_new_dense();
        d.values_mut().copy_from_slice(&[0.5, f64::NAN, 2e7]);
        let v: Rc<dyn Vector> = Rc::from(d);
        let out = clone_clamped(&v, 1e-3, 1e6, 7.0);
        let out = out.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(out.values()[0], 0.5);
        assert_eq!(out.values()[1], 7.0); // unseeded -> fill
        assert_eq!(out.values()[2], 1e6); // then the cap applies
    }

    /// The re-optimize path reuses the previous solve's vector spaces,
    /// which are compound for a blocked NLP. NaN has to resolve there
    /// too: a debug-only guard is compiled out of the release builds
    /// that ship, so an unresolved NaN would ride the clamps into the
    /// iterate and poison the solve.
    #[test]
    fn nan_resolves_inside_a_compound_vector() {
        let inner = DenseVectorSpace::new(2);
        let space = CompoundVectorSpace::new(2, 4);
        for icomp in 0..2 {
            let inner = Rc::clone(&inner);
            space.set_comp(icomp, 2, move || {
                let mut d = inner.make_new_dense();
                d.set(0.0);
                Box::new(d)
            });
        }
        let mut cv = CompoundVector::new(Rc::clone(&space));
        for (icomp, vals) in [[0.5, f64::NAN], [f64::NAN, 2e7]].into_iter().enumerate() {
            let c = cv.comp_mut(icomp as pounce_common::types::Index);
            let d = c.as_any_mut().downcast_mut::<DenseVector>().unwrap();
            d.values_mut().copy_from_slice(&vals);
        }

        let v: Rc<dyn Vector> = Rc::from(cv);
        let out = clone_clamped(&v, 1e-3, 1e6, 7.0);

        let out = out.as_any().downcast_ref::<CompoundVector>().unwrap();
        let flat: Vec<f64> = (0..out.n_comps())
            .flat_map(|i| {
                out.comp(i)
                    .as_any()
                    .downcast_ref::<DenseVector>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(flat[0], 0.5);
        assert_eq!(flat[1], 7.0); // unseeded -> fill, not NaN
        assert_eq!(flat[2], 7.0);
        assert_eq!(flat[3], 1e6); // then the cap applies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    fn dense(n: i32, fill: f64) -> Rc<dyn Vector> {
        let space = DenseVectorSpace::new(n);
        let mut v = space.make_new_dense();
        v.set(fill);
        Rc::new(v)
    }

    #[test]
    fn clamps_multipliers_to_cap() {
        let v = dense(3, 1e10);
        let out = clone_clamped(&v, 0.0, 1e6, 0.0);
        assert_eq!(out.amax(), 1e6);
        let v2 = dense(3, -1e10);
        let out2 = clone_clamped(&v2, -1e6, 1e6, 0.0);
        assert_eq!(out2.amax(), 1e6);
    }

    #[test]
    fn clamps_bound_mults_nonneg() {
        let v = dense(3, -5.0);
        let out = clone_clamped(&v, 0.0, 1e6, 0.0);
        assert_eq!(out.amax(), 0.0);
    }

    #[test]
    fn empty_vector_short_circuits() {
        let v = dense(0, 0.0);
        let out = clone_clamped(&v, 0.0, 1.0, 0.0);
        assert_eq!(out.dim(), 0);
    }

    #[test]
    fn in_range_values_pass_through_untouched() {
        let v = dense(3, 0.5);
        let out = clone_clamped(&v, 0.0, 1.0, 0.0);
        assert!((out.max() - 0.5).abs() < 1e-15);
        assert!((out.min() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn mult_bound_push_floors_zero_bound_multipliers() {
        // A carried-in z = 0 (inactive bound in the previous solution)
        // must be floored at warm_start_mult_bound_push, matching
        // upstream's ElementWiseMax — the barrier needs z > 0.
        let v = dense(3, 0.0);
        let out = clone_clamped(&v, 1e-3, 1e6, 0.0);
        assert!((out.min() - 1e-3).abs() < 1e-18);
        // Values already above the floor pass through.
        let v2 = dense(3, 0.7);
        let out2 = clone_clamped(&v2, 1e-3, 1e6, 0.0);
        assert!((out2.max() - 0.7).abs() < 1e-15);
    }

    #[test]
    fn uninitialized_source_collapses_to_zero() {
        // Application's placeholder seed iterate: vector allocated but
        // never written. `clone_clamped` must fall back to zero instead
        // of tripping the dense-vector "must be initialized" assert.
        let space = DenseVectorSpace::new(4);
        let v: Rc<dyn Vector> = Rc::new(space.make_new_dense());
        let out = clone_clamped(&v, 0.0, 1e6, 0.0);
        assert_eq!(out.amax(), 0.0);
    }
}

/// Did the caller supply *any* dual information?
///
/// True when some equality multiplier is non-zero, or some bound
/// multiplier is strictly positive and finite. Both are the
/// signatures the per-block "unseeded" tests use, taken across every
/// block at once: an entry that is exactly `0` cannot have come from a
/// converged solve, and neither can a whole `y` block of zeros.
///
/// This is the gate on the reconstruction as a whole. Completing a
/// partial warm start is well-posed — the supplied blocks pin the
/// missing ones through stationarity. Manufacturing all of them from a
/// primal point alone is not: the result is the cold path's estimate
/// paired with the warm path's barrier, and it measured worse than the
/// constants it replaced.
fn any_dual_seeded(data: &IpoptDataHandle) -> bool {
    let Some(curr) = data.borrow().curr.clone() else {
        return false;
    };
    for y in [&curr.y_c, &curr.y_d] {
        if y.dim() > 0 && !is_identically_zero(y) {
            return true;
        }
    }
    for z in [&curr.z_l, &curr.z_u, &curr.v_l, &curr.v_u] {
        if z.dim() == 0 {
            continue;
        }
        if let Some(vals) = flatten(&**z) {
            if vals.iter().any(|v| v.is_finite() && *v > 0.0) {
                return true;
            }
        }
    }
    false
}
