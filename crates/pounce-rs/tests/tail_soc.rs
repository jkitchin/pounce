//! gh#1211 — the tail second-order correction (`tail_soc`).
//!
//! Once the monotone μ is at its floor, a full step accepted on its first
//! trial leaves a constraint residual `c(x + d) = O(‖d‖²)`, and the ordinary
//! SOC (`max_soc`) never runs on it because it is tried only after a rejected
//! first trial. With `tail_soc=yes` such a step is replaced by its
//! second-order correction when the corrected point is a full step, lowers
//! the constraint violation, and passes the same filter test. Accepted
//! corrections are tagged with an upper-case line-search character.
//!
//! The model is a small direct-collocation problem of the class the issue
//! was reported on (a minimum-lap-time NLP, 73k–127k variables): trapezoidal
//! collocation of `x' = sin x + u` with `|u| ≤ 1.2` and a terminal penalty.
//! Its constraints are curved, so a full step leaves a second-order residual
//! for the correction to remove. It is far too small to show the tail
//! reduction the issue measured; it pins the mechanics:
//!
//! * the option is off by default and then never acts;
//! * with it on, corrections fire, and only once μ is at its floor;
//! * the answer does not move and the iteration count does not grow.
//!
//! `max_soc=0` so every upper-case character in the log is this option's.

use pounce_rs::builder::{Nlp, Problem};

const N: usize = 40;
const T: f64 = 2.0;
const TARGET: f64 = 2.0;
const PENALTY: f64 = 10.0;

/// Variables `[x_0 … x_N, u_0 … u_{N-1}]`; constraint `k` is the trapezoid
/// rule on interval `k` with `u` piecewise constant:
/// `x_{k+1} − x_k − h/2·(sin x_k + sin x_{k+1}) − h·u_k = 0`.
struct Collocation;

impl Collocation {
    const H: f64 = T / N as f64;
    const NV: usize = 2 * N + 1;
    fn u(k: usize) -> usize {
        N + 1 + k
    }
}

impl Problem for Collocation {
    fn objective(&self, v: &[f64]) -> f64 {
        let effort: f64 = (0..N).map(|k| v[Self::u(k)].powi(2)).sum::<f64>() * Self::H;
        effort + PENALTY * (v[N] - TARGET).powi(2)
    }
    fn gradient(&self, v: &[f64], g: &mut [f64]) -> bool {
        g.fill(0.0);
        for k in 0..N {
            g[Self::u(k)] = 2.0 * Self::H * v[Self::u(k)];
        }
        g[N] = 2.0 * PENALTY * (v[N] - TARGET);
        true
    }
    fn n_constraints(&self) -> usize {
        N
    }
    fn constraints(&self, v: &[f64], out: &mut [f64]) {
        let h = Self::H;
        for k in 0..N {
            out[k] = v[k + 1] - v[k] - 0.5 * h * (v[k].sin() + v[k + 1].sin()) - h * v[Self::u(k)];
        }
    }
    fn jacobian(&self, v: &[f64], jac: &mut [f64]) -> bool {
        let h = Self::H;
        jac.fill(0.0);
        for k in 0..N {
            let row = &mut jac[k * Self::NV..(k + 1) * Self::NV];
            row[k] = -1.0 - 0.5 * h * v[k].cos();
            row[k + 1] = 1.0 - 0.5 * h * v[k + 1].cos();
            row[Self::u(k)] = -h;
        }
        true
    }
}

struct Run {
    success: bool,
    iters: i32,
    objective: f64,
    /// Iterations whose line-search tag is upper case.
    corrected: Vec<i32>,
    /// First iteration whose μ is the final one.
    first_at_final_mu: i32,
}

fn solve(tail_soc: Option<&str>) -> Run {
    let nv = Collocation::NV;
    let mut lo = vec![-1e19; nv];
    let mut hi = vec![1e19; nv];
    lo[0] = 0.0;
    hi[0] = 0.0;
    for k in 0..N {
        lo[Collocation::u(k)] = -1.2;
        hi[Collocation::u(k)] = 1.2;
    }
    let mut nlp = Nlp::new(Collocation)
        .x0(&vec![0.0; nv])
        .var_bounds(&lo, &hi)
        .constraint_bounds(&vec![0.0; N], &vec![0.0; N])
        .option_str("hessian_approximation", "limited-memory")
        .option_str("mu_strategy", "monotone")
        .option_int("max_soc", 0)
        .option_num("tol", 1e-4)
        .option_int("print_level", 0)
        .capture_iterations();
    if let Some(v) = tail_soc {
        nlp = nlp.option_str("tail_soc", v);
    }
    let sol = nlp.solve();
    let rows = &sol.stats.iterations;
    let final_mu = rows.last().map_or(f64::NAN, |r| r.mu);
    Run {
        success: sol.success,
        iters: sol.stats.iteration_count,
        objective: sol.objective,
        corrected: rows
            .iter()
            .filter(|r| r.alpha_primal_char.is_ascii_uppercase())
            .map(|r| r.iter)
            .collect(),
        first_at_final_mu: rows
            .iter()
            .find(|r| r.mu == final_mu)
            .map_or(i32::MAX, |r| r.iter),
    }
}

#[test]
fn off_by_default_and_then_inert() {
    let default = solve(None);
    let off = solve(Some("no"));
    assert!(default.success && off.success);
    assert!(default.corrected.is_empty(), "{:?}", default.corrected);
    assert!(off.corrected.is_empty(), "{:?}", off.corrected);
    assert_eq!(default.iters, off.iters);
    assert_eq!(default.objective, off.objective);
}

#[test]
fn corrections_fire_only_at_the_final_mu_and_do_not_cost_the_answer() {
    let off = solve(Some("no"));
    let on = solve(Some("yes"));
    assert!(off.success && on.success);
    assert!(
        !on.corrected.is_empty(),
        "the model must exercise the correction, or this test says nothing"
    );
    // The gate: nothing before μ reaches its floor. A row's μ is the one the
    // step was computed at, so the first corrected row is no earlier than
    // the first row printed at the final μ.
    assert!(
        on.corrected.iter().all(|&k| k >= on.first_at_final_mu),
        "corrected {:?} before μ reached its floor at {}",
        on.corrected,
        on.first_at_final_mu
    );
    assert!(on.iters <= off.iters, "on {} > off {}", on.iters, off.iters);
    let rel = (on.objective - off.objective).abs() / off.objective.abs().max(1.0);
    assert!(rel < 1e-6, "objective moved by {rel:e}");
}
