//! Dump the Phase-6 plan for an instance fed as JSON-ish text on stdin
//! (adversary debugging tool; reads the same arrays the probe writes into `.nl`).
//!
//! stdin format, one item per line:
//!   n
//!   rows      : "col:coef,col:coef;col:coef,..."   (rows separated by ';')
//!   row_const : comma-separated
//!   g         : comma-separated
//!   x_l       : comma-separated
//!   x_u       : comma-separated
use pounce_presolve::{PlanConfig, PlanInput, VarRecovery, build_plan};
use std::io::Read;

fn nums(s: &str) -> Vec<f64> {
    s.trim().split(',').filter(|t| !t.is_empty()).map(|t| t.trim().parse().unwrap()).collect()
}

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut it = buf.lines();
    let n: usize = it.next().unwrap().trim().parse().unwrap();
    let rows: Vec<Vec<(usize, f64)>> = it
        .next().unwrap().trim().split(';').filter(|s| !s.is_empty())
        .map(|r| r.split(',').filter(|s| !s.is_empty()).map(|e| {
            let (c, a) = e.split_once(':').unwrap();
            (c.trim().parse().unwrap(), a.trim().parse().unwrap())
        }).collect())
        .collect();
    let row_const = nums(it.next().unwrap());
    let g = nums(it.next().unwrap());
    let x_l = nums(it.next().unwrap());
    let x_u = nums(it.next().unwrap());
    let m = rows.len();
    let eligible = vec![true; m];
    let plan = build_plan(
        &PlanInput { n_vars: n, n_rows: m, rows: &rows, row_const: &row_const,
                     g_l: &g, g_u: &g, eligible: &eligible, x_l: &x_l, x_u: &x_u },
        &PlanConfig::default(),
    );
    let mut collapsed = Vec::new();
    for (red, &k) in plan.vars_kept.iter().enumerate() {
        let (lo, hi) = (plan.x_l_red[red], plan.x_u_red[red]);
        let pointlike = lo.is_finite() && hi.is_finite()
            && (hi - lo).abs() <= 1e-9 * lo.abs().max(hi.abs()).max(1.0);
        let declared_point = (x_u[k] - x_l[k]).abs() <= 1e-12;
        if pointlike && !declared_point {
            collapsed.push(k);
        }
    }
    println!("kept={:?} steps={} collapsed_reduced_boxes={:?}",
             plan.vars_kept, plan.steps.len(), collapsed);
    for (red, &k) in plan.vars_kept.iter().enumerate() {
        println!("  col{k}: reduced=[{:.12}, {:.12}] src=({},{}) declared=[{}, {}]",
                 plan.x_l_red[red], plan.x_u_red[red],
                 plan.x_l_src[red], plan.x_u_src[red], x_l[k], x_u[k]);
    }
    for (j, r) in plan.recovery.iter().enumerate() {
        if !matches!(r, VarRecovery::Kept(_)) { println!("  col{j} -> {r:?}"); }
    }
}
