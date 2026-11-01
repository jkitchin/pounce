//! Centrality-floored fraction-to-the-boundary (pounce gh#1098).
//!
//! Ipopt's rule `α = τ·α_bnd`, `τ = max(τ_min, 1 − μ)`, lets the slack or
//! bound multiplier that blocks a step land at `(1 − τ) ≈ μ` of its value
//! once `μ` is small. A blocking pair that sat on the central path
//! (`s·v ≈ μ`) is then knocked to `s·v ≈ μ²`, six orders off it at
//! `μ = 1.8e-6`, and when the next direction reverses the step the solver
//! spends 5–10 iterations recentering it. On Mittelmann `qcqp1500-1nc`
//! that cycle is the whole of the crawl: a walk from vertex to vertex of a
//! nonconvex polytope at ~15 iterations per net change of active set, for
//! ~1200 iterations at a fixed `μ` (Ipopt 3.14 crawls identically).
//!
//! The remedy is Mehrotra's step-length heuristic (Mehrotra 1992 §5;
//! Wright, *Primal-Dual Interior-Point Methods*, eq. (10.9)): keep a
//! `1 − γ` floor on the fraction, but let the step approach the boundary
//! beyond it only while the *blocking pair's* complementarity product
//! stays `≥ γ·μ`, so a variable driven to its bound lands near its central
//! value instead of at `μ` of itself.
//!
//! It is applied only to a blocking pair that is currently centered
//! (`s·v ≥ γ·μ`). The rule presumes a neighbourhood of the central path;
//! a pair already far below it has no centrality to preserve, and holding
//! its step at `1 − γ` only slows its way to zero. Measured on Mittelmann
//! `clnlbeam` (a bang-bang switching point migrating across 39 998
//! doubly-bounded variables whose blockers sit at `s·v/μ ≈ 1e-3…1e-6`):
//! the uncentered rule hit `max_iter` where the default converges.
//!
//! The floored step is capped at the τ rule's step, so the floor only ever
//! *shortens* a step. Uncapped, a pair far above the central path early on
//! (`s·v = 9` at `μ = 0.1`) was allowed a longer step than τ and landed at
//! `γμ`, less central than upstream leaves it; that flipped
//! `issue_390_nonlinear_equality_scale` at row scale 1e-8 from an
//! infeasibility certificate to `Restoration_Failed`.
//!
//! **Off by default** (`γ = 0` restores upstream's rule exactly). Even
//! capped, at `γ = 0.01` it is the only setting measured that solves
//! Mittelmann `qcqp1500-1c` (148 iterations; main and the default both hit
//! the CPU limit), but it also turns the `discs` fixture into a false
//! `Infeasible_Problem_Detected` on the exact leg and changes the verdict
//! of three purpose-built integration tests (`issue_884`,
//! `masked_certificate_fuzz` with the certificate veto off,
//! `line_search_options_wiring`). The measured opt-in for gh#1098 is
//! `perturb_slack_fact=0.03` alone.

use crate::ipopt_cq::IpoptCqHandle;
use crate::iterates_vector::IteratesVector;
use pounce_common::types::Number;
use pounce_linalg::Vector;

fn dense(v: &dyn Vector) -> Option<Vec<Number>> {
    v.as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
        .map(|d| d.expanded_values())
}

/// Registered default of `ftb_centrality_floor` (γ).
pub const DEFAULT_FTB_CENTRALITY_FLOOR: Number = 0.0;

/// Registered default of `perturb_slack_fact` (`δ_s = fact · δ_x`). `1` is
/// upstream Ipopt's tie; `0.03` solves gh#1098 but is opt-in, see the
/// option's registration and `dev-notes/qcqp1500-crawl-slack-perturbation.md`.
pub const DEFAULT_PERTURB_SLACK_FACT: Number = 1.0;

/// Largest fraction of the step to the boundary the floor may take.
/// Mathematically `f < 1` always (it is `1 − γμ/(s·v_new)`), but it rounds
/// to `1` when the pair's product dwarfs `γμ`, which would put the blocking
/// variable exactly on its bound.
const MAX_FRACTION: Number = 1.0 - 1e-12;

/// Returns `Some((alpha_p, alpha_d))` or `None` when a block is not a dense
/// vector (restoration's compound iterates); the caller then keeps the
/// ordinary τ rule.
pub fn floored_alphas(
    cq: &IpoptCqHandle,
    delta: &IteratesVector,
    mu: Number,
    gamma: Number,
    tau: Number,
) -> Option<(Number, Number)> {
    let cq = cq.borrow();
    let iv = cq.curr_iv();
    let nlp = cq.nlp().borrow();

    // Slack blocks (4) with their projected steps and paired multipliers.
    let s_x_l = cq.curr_slack_x_l();
    let s_x_u = cq.curr_slack_x_u();
    let s_s_l = cq.curr_slack_s_l();
    let s_s_u = cq.curr_slack_s_u();
    let mut d_x_l = s_x_l.make_new();
    nlp.px_l()
        .trans_mult_vector(1.0, &*delta.x, 0.0, &mut *d_x_l);
    let mut d_x_u = s_x_u.make_new();
    nlp.px_u()
        .trans_mult_vector(-1.0, &*delta.x, 0.0, &mut *d_x_u);
    let mut d_s_l = s_s_l.make_new();
    nlp.pd_l()
        .trans_mult_vector(1.0, &*delta.s, 0.0, &mut *d_s_l);
    let mut d_s_u = s_s_u.make_new();
    nlp.pd_u()
        .trans_mult_vector(-1.0, &*delta.s, 0.0, &mut *d_s_u);

    let blocks: [(&dyn Vector, &dyn Vector, &dyn Vector, &dyn Vector); 4] = [
        (&*s_x_l, &*d_x_l, &*iv.z_l, &*delta.z_l),
        (&*s_x_u, &*d_x_u, &*iv.z_u, &*delta.z_u),
        (&*s_s_l, &*d_s_l, &*iv.v_l, &*delta.v_l),
        (&*s_s_u, &*d_s_u, &*iv.v_u, &*delta.v_u),
    ];
    let mut dense_blocks = Vec::with_capacity(4);
    for (s, ds, v, dv) in blocks {
        if s.dim() == 0 {
            continue;
        }
        dense_blocks.push((dense(s)?, dense(ds)?, dense(v)?, dense(dv)?));
    }

    Some(floored_from_blocks(&dense_blocks, mu, gamma, tau))
}

/// One complementarity block: slack `s`, its step `ds`, multiplier `v`,
/// its step `dv`.
pub(crate) type Block = (Vec<Number>, Vec<Number>, Vec<Number>, Vec<Number>);

/// The step-length rule itself, on dense blocks.
pub(crate) fn floored_from_blocks(
    dense_blocks: &[Block],
    mu: Number,
    gamma: Number,
    tau: Number,
) -> (Number, Number) {
    // Pure step to the boundary (τ = 1) and the blocking pair on each side.
    let mut ap_bnd = 1.0_f64;
    let mut ad_bnd = 1.0_f64;
    let mut ip: Option<(usize, usize)> = None;
    let mut id: Option<(usize, usize)> = None;
    for (b, (s, ds, v, dv)) in dense_blocks.iter().enumerate() {
        for i in 0..s.len() {
            if ds[i] < 0.0 {
                let a = -s[i] / ds[i];
                if a < ap_bnd {
                    ap_bnd = a;
                    ip = Some((b, i));
                }
            }
            if dv[i] < 0.0 {
                let a = -v[i] / dv[i];
                if a < ad_bnd {
                    ad_bnd = a;
                    id = Some((b, i));
                }
            }
        }
    }
    let floor = 1.0 - gamma;
    // A pair already below γμ is not on the central path: keep the τ rule.
    let uncentered = |b: usize, i: usize| dense_blocks[b].0[i] * dense_blocks[b].2[i] < gamma * mu;

    // Primal: the blocking slack may approach its bound only while its
    // product with the (fully dual-stepped) multiplier stays ≥ γμ.
    let alpha_p = match ip {
        None => 1.0,
        Some((b, i)) if uncentered(b, i) => (tau * ap_bnd).min(1.0),
        Some((b, i)) => {
            let (s, ds, v, dv) = &dense_blocks[b];
            let v_new = v[i] + ad_bnd * dv[i];
            let f = if v_new > 0.0 {
                (gamma * mu / v_new - s[i]) / (ap_bnd * ds[i])
            } else {
                floor
            };
            (floor.max(f).min(tau).min(MAX_FRACTION) * ap_bnd).min(1.0)
        }
    };
    let alpha_d = match id {
        None => 1.0,
        Some((b, i)) if uncentered(b, i) => (tau * ad_bnd).min(1.0),
        Some((b, i)) => {
            let (s, ds, v, dv) = &dense_blocks[b];
            let s_new = s[i] + alpha_p * ds[i];
            let f = if s_new > 0.0 {
                (gamma * mu / s_new - v[i]) / (ad_bnd * dv[i])
            } else {
                floor
            };
            (floor.max(f).min(tau).min(MAX_FRACTION) * ad_bnd).min(1.0)
        }
    };
    (alpha_p, alpha_d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(s: &[f64], ds: &[f64], v: &[f64], dv: &[f64]) -> Block {
        (s.to_vec(), ds.to_vec(), v.to_vec(), dv.to_vec())
    }

    const TAU: f64 = 0.99;

    #[test]
    fn no_blocking_variable_takes_full_step() {
        let b = [blk(&[1.0], &[1.0], &[1.0], &[1.0])];
        assert_eq!(floored_from_blocks(&b, 1e-6, 0.01, TAU), (1.0, 1.0));
    }

    #[test]
    fn uncentered_blocker_keeps_tau_rule() {
        // s·v = 1e-12 < γμ = 1e-8: no centrality to preserve.
        let mu = 1e-6;
        let b = [blk(&[1e-6], &[-2e-6], &[1e-6], &[-4e-6])];
        let (ap, ad) = floored_from_blocks(&b, mu, 0.01, TAU);
        assert!((ap - TAU * 0.5).abs() < 1e-15);
        assert!((ad - TAU * 0.25).abs() < 1e-15);
    }

    #[test]
    fn centered_blocker_stays_at_least_gamma_mu() {
        // Centered pair s·v = μ, primal blocker only: under τ it would land
        // at s·v ≈ (1−τ)·μ·(v_new/v); the floor keeps it ≥ γμ.
        let (mu, gamma) = (1e-6, 0.01);
        let b = [blk(&[1e-3], &[-2e-3], &[1e-3], &[0.0])];
        let (ap, ad) = floored_from_blocks(&b, mu, gamma, 1.0 - mu);
        assert_eq!(ad, 1.0);
        let (s, ds, v, _) = &b[0];
        let prod = (s[0] + ap * ds[0]) * v[0];
        assert!(prod >= gamma * mu * (1.0 - 1e-9), "prod {prod}");
        // α_bnd = 0.5; the floored step lies in [(1−γ)·α_bnd, α_bnd).
        assert!(ap >= (1.0 - gamma) * 0.5 && ap < 0.5);
        // The τ rule at this μ would have crushed it to ~μ².
        let prod_tau = (s[0] + (1.0 - mu) * 0.5 * ds[0]) * v[0];
        assert!(prod_tau < 1e-11);
    }

    #[test]
    fn floor_bounds_the_fraction_from_below() {
        // Multiplier collapses alongside the slack, so keeping s·v ≥ γμ
        // would need a tiny step; the 1−γ floor wins.
        let (mu, gamma) = (1e-6, 0.01);
        let b = [blk(&[1e-3], &[-2e-3], &[1e-3], &[-0.999e-3])];
        let (ap, _) = floored_from_blocks(&b, mu, gamma, TAU);
        assert!((ap - (1.0 - gamma) * 0.5).abs() < 1e-12, "ap {ap}");
    }

    #[test]
    fn fraction_never_rounds_onto_the_bound() {
        // s·v ≫ γμ makes f round to 1 in floating point.
        let b = [blk(&[1.0], &[-2.0], &[1e6], &[0.0])];
        let (ap, _) = floored_from_blocks(&b, 1e-12, 0.01, TAU);
        assert!(ap < 0.5);
        assert!(1.0 + ap * -2.0 > 0.0);
    }

    #[test]
    fn never_steps_further_than_the_tau_rule() {
        // gh#1098 / issue_390: a pair far above the central path (s·v = 9,
        // μ = 0.1). Mechanically "keep s·v ≥ γμ" would allow a longer step
        // than τ and land the multiplier at γμ/s, i.e. less central than the
        // τ rule leaves it. The floor only ever shortens.
        let b = [blk(&[9.0], &[0.0], &[1.0], &[-1.09])];
        let (_, ad) = floored_from_blocks(&b, 0.1, 0.01, TAU);
        assert!((ad - TAU / 1.09).abs() < 1e-15, "ad {ad}");
    }
}
