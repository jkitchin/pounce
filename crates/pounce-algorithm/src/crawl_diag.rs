//! Env-gated (`POUNCE_CRAWL_DIAG`) per-iteration diagnostic for gh#1098:
//! which slack row caps the primal/dual fraction-to-the-boundary step,
//! what carries `inf_pr`, and how the step splits between x and s.
//! Prototype instrumentation; not wired to any option.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::iterates_vector::IteratesVector;
use pounce_common::types::Number;
use pounce_linalg::Vector;

fn dense(v: &dyn Vector) -> Vec<Number> {
    v.as_any()
        .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_default()
}

pub fn report(
    data: &IpoptDataHandle,
    cq: &IpoptCqHandle,
    delta: &IteratesVector,
    tau: Number,
    alpha_p_max: Number,
    alpha_d_max: Number,
) {
    let (iter, mu, curr) = {
        let d = data.borrow();
        let Some(c) = d.curr.clone() else { return };
        (d.iter_count, d.curr_mu, c)
    };
    let cq = cq.borrow();
    let sl = dense(&*cq.curr_slack_s_l());
    let vl = dense(&*curr.v_l);
    let dvl = dense(&*delta.v_l);
    let ds_proj = {
        let nlp = cq.nlp().borrow();
        let mut t = cq.curr_slack_s_l().make_new();
        nlp.pd_l().trans_mult_vector(1.0, &*delta.s, 0.0, &mut *t);
        dense(&*t)
    };
    let dms = dense(&*cq.curr_d_minus_s());
    let dx = dense(&*delta.x);
    if let Some(dir) = std::env::var_os("POUNCE_CRAWL_DUMP") {
        let dir = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let write = |name: &str, v: &[Number]| {
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            let _ = std::fs::write(dir.join(format!("{name}_{iter}.f64")), bytes);
        };
        write("x", &dense(&*curr.x));
        write("dx", &dx);
        write("sl", &sl);
        write("vl", &vl);
        write("dsl", &ds_proj);
        write("dvl", &dvl);
        write("gl", &dense(&*cq.curr_grad_lag_x()));
        write("yd", &dense(&*curr.y_d));
        write("dms", &dms);
    }
    let ds = dense(&*delta.s);
    let dx_inf = dx.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let dx_2 = dx.iter().map(|b| b * b).sum::<f64>().sqrt();
    let ds_inf = ds.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    // primal blocker
    let mut best = (Number::INFINITY, usize::MAX);
    for i in 0..sl.len() {
        if ds_proj[i] < 0.0 {
            let a = -tau * sl[i] / ds_proj[i];
            if a < best.0 {
                best = (a, i);
            }
        }
    }
    let mut bestd = (Number::INFINITY, usize::MAX);
    for i in 0..vl.len() {
        if dvl[i] < 0.0 {
            let a = -tau * vl[i] / dvl[i];
            if a < bestd.0 {
                bestd = (a, i);
            }
        }
    }
    let mut imax = 0;
    for i in 0..dms.len() {
        if dms[i].abs() > dms[imax].abs() {
            imax = i;
        }
    }
    let n_cut = (0..sl.len())
        .filter(|&i| ds_proj[i] < 0.0 && -tau * sl[i] / ds_proj[i] < 1.0)
        .count();
    let n_act = (0..sl.len())
        .filter(|&i| vl[i] * vl[i] / mu > 100.0)
        .count();
    let pb = |i: usize| -> String {
        if i == usize::MAX {
            return "-".into();
        }
        format!(
            "row={} s={:.2e} v={:.2e} ds={:.2e} sv/mu={:.1e}",
            i,
            sl[i],
            vl[i],
            ds_proj[i],
            sl[i] * vl[i] / mu
        )
    };
    let db = |i: usize| -> String {
        if i == usize::MAX {
            return "-".into();
        }
        format!(
            "row={} s={:.2e} v={:.2e} dv={:.2e} sv/mu={:.1e}",
            i,
            sl[i],
            vl[i],
            dvl[i],
            sl[i] * vl[i] / mu
        )
    };
    eprintln!(
        "CRAWL it={} mu={:.2e} aPmax={:.3e} aDmax={:.3e} |dx|inf={:.2e} |dx|2={:.2e} |ds|inf={:.2e} ncut={} nact={} | P: {} | D: {} | infpr: row={} d-s={:.2e} s={:.2e} v={:.2e}",
        iter,
        mu,
        alpha_p_max,
        alpha_d_max,
        dx_inf,
        dx_2,
        ds_inf,
        n_cut,
        n_act,
        pb(best.1),
        db(bestd.1),
        imax,
        dms.get(imax).copied().unwrap_or(f64::NAN),
        sl.get(imax).copied().unwrap_or(f64::NAN),
        vl.get(imax).copied().unwrap_or(f64::NAN),
    );
}
