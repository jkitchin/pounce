//! gh#1098 prototype: fraction-to-the-boundary with a centrality floor
//! (Mehrotra 1992 §5; Wright, *Primal-Dual Interior-Point Methods*, eq.
//! (10.9)). Env-gated by `POUNCE_FTB_FLOOR=<gamma>`; default behaviour is
//! unchanged when the variable is unset.
//!
//! Ipopt's rule τ = max(τ_min, 1 − μ) lets the blocking slack or bound
//! multiplier land at (1 − τ) ≈ μ of its value once μ is small — six orders
//! off the central path at μ = 1.8e-6 — and every such cut costs a long
//! recentering when the step that caused it is reversed by the next one.
//! This rule keeps the τ_min = 1 − γ floor but lets the step approach the
//! boundary beyond it only while the *blocking pair's* complementarity
//! product stays ≥ γ·μ, so a variable driven to its bound lands near its
//! central value instead of at μ of itself.

use crate::ipopt_cq::IpoptCqHandle;
use crate::iterates_vector::IteratesVector;
use pounce_common::types::Number;
use pounce_linalg::Vector;

fn dense(v: &dyn Vector) -> Option<Vec<Number>> {
    v.as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
        .map(|d| d.expanded_values())
}

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
    // `POUNCE_FTB_FLOOR_CENTERED`: the rule presumes the blocking pair is
    // near the central path. When it already sits below γμ there is no
    // centrality to preserve, and holding τ at 1 − γ only slows the slack's
    // way to zero (clnlbeam: switching-point migration along a bang-bang
    // arc, blockers at s·v/μ ≈ 1e-3…1e-6). Fall back to the τ rule there.
    let only_centered = std::env::var_os("POUNCE_FTB_FLOOR_CENTERED").is_some();

    // Primal: the blocking slack may approach its bound only while its
    // product with the (fully dual-stepped) multiplier stays ≥ γμ.
    let alpha_p = match ip {
        None => 1.0,
        Some((b, i))
            if only_centered && dense_blocks[b].0[i] * dense_blocks[b].2[i] < gamma * mu =>
        {
            (tau * ap_bnd).min(1.0)
        }
        Some((b, i)) => {
            let (s, ds, v, dv) = &dense_blocks[b];
            let v_new = v[i] + ad_bnd * dv[i];
            let f = if v_new > 0.0 {
                (gamma * mu / v_new - s[i]) / (ap_bnd * ds[i])
            } else {
                floor
            };
            (floor.max(f).min(1.0) * ap_bnd).min(1.0)
        }
    };
    let alpha_d = match id {
        None => 1.0,
        Some((b, i))
            if only_centered && dense_blocks[b].0[i] * dense_blocks[b].2[i] < gamma * mu =>
        {
            (tau * ad_bnd).min(1.0)
        }
        Some((b, i)) => {
            let (s, ds, v, dv) = &dense_blocks[b];
            let s_new = s[i] + alpha_p * ds[i];
            let f = if s_new > 0.0 {
                (gamma * mu / s_new - v[i]) / (ad_bnd * dv[i])
            } else {
                floor
            };
            (floor.max(f).min(1.0) * ad_bnd).min(1.0)
        }
    };
    if std::env::var_os("POUNCE_FTB_FLOOR_DIAG").is_some() {
        let names = ["x_l", "x_u", "s_l", "s_u"];
        let show = |t: Option<(usize, usize)>| -> String {
            match t {
                None => "-".into(),
                Some((b, i)) => {
                    let (s, ds, v, dv) = &dense_blocks[b];
                    format!(
                        "{}[{}] s={:.2e} ds={:.2e} v={:.2e} dv={:.2e} sv/mu={:.1e}",
                        names[b],
                        i,
                        s[i],
                        ds[i],
                        v[i],
                        dv[i],
                        s[i] * v[i] / mu
                    )
                }
            }
        };
        eprintln!(
            "FTB mu={:.2e} ap_bnd={:.3e} aP={:.3e} ad_bnd={:.3e} aD={:.3e} | P {} | D {}",
            mu,
            ap_bnd,
            alpha_p,
            ad_bnd,
            alpha_d,
            show(ip),
            show(id)
        );
    }
    Some((alpha_p, alpha_d))
}
