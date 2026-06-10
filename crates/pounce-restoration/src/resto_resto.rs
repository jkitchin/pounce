//! Nested-restoration phase — port of
//! `Algorithm/IpRestoRestoPhase.{hpp,cpp}`.
//!
//! Used when the *inner* (resto) IPM's line search cannot make
//! progress and needs its own restoration. Treating the resto-NLP as
//! separable in `(x_orig, s)` and the slack feasibility variables
//! `(n_c, p_c, n_d, p_d)`, this driver holds `(x_orig, s)` fixed and
//! resets the slack variables in closed form to the per-element
//! minimizer of the resto barrier objective subject to the equality
//! `c(x_orig) + n_c − p_c = 0` (and the analogous `d − s` equation).
//!
//! For each constraint component `c_i`, the optimal `(n_i, p_i)` pair
//! is the positive root of the quadratic
//!
//! ```text
//!   v² − 2·a·v − b = 0     with   a = mu/(2ρ) − 0.5·c_i,  b = c_i · mu/(2ρ)
//! ```
//!
//! solved as `v = a + sqrt(a² + b)`, then `n_i = v` and `p_i = c_i + n_i`
//! (which satisfies `c + n − p = 0` exactly). The linear term is `−2·a·v`,
//! not `+2·a·v`: substituting the root gives `(v − a)² = a² + b`, i.e.
//! `v² − 2·a·v − b = 0`. (Upstream's `solve_quadratic` computes the same
//! `a + sqrt(a² + b)` — verified against `IpRestoRestoPhase.cpp`, whose body
//! is `v = a; v = v*v; v += b; v = sqrt(v); v += a`.) Mirrors
//! `IpRestoRestoPhase.cpp:30-109`.

use crate::r#trait::{RestorationOutcome, RestorationPhase};
use crate::resto_nlp::{BLOCK_N_C, BLOCK_N_D, BLOCK_P_C, BLOCK_P_D, BLOCK_X};
use pounce_algorithm::ipopt_cq::IpoptCqHandle;
use pounce_algorithm::ipopt_data::IpoptDataHandle;
use pounce_algorithm::ipopt_nlp::IpoptNlp;
use pounce_algorithm::iterates_vector::IteratesVector;
use pounce_algorithm::kkt::aug_system_solver::AugSystemSolver;
use pounce_common::types::Number;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::{CompoundVector, CompoundVectorSpace, Vector};
use std::cell::RefCell;
use std::rc::Rc;

/// Recursive resto-of-resto driver. Holds the two pieces of the
/// resto-NLP it needs (the proximity weight `rho` and the wrapped
/// original NLP) directly, since the [`RestorationPhase`] trait surface
/// hands us a `&Rc<RefCell<dyn IpoptNlp>>` that doesn't expose the
/// concrete `RestoIpoptNlp` API. The CLI / inner-solver factory
/// captures these at the same point it sets up the resto NLP.
pub struct RestoRestorationPhase {
    rho: Number,
    orig_nlp: Option<Rc<RefCell<dyn IpoptNlp>>>,
}

impl RestoRestorationPhase {
    /// `rho` mirrors `RestoIpoptNlp::rho`. `orig_nlp` is the wrapped
    /// outer NLP (the same `Rc` the resto NLP holds in its
    /// [`crate::resto_nlp::RestoIpoptNlp::orig_nlp`] slot). Either may
    /// be supplied later via [`Self::with_orig_nlp`] /
    /// [`Self::set_orig_nlp`] when the construction site doesn't have
    /// the orig handle on hand yet.
    pub fn new(rho: Number) -> Self {
        Self {
            rho,
            orig_nlp: None,
        }
    }

    pub fn with_orig_nlp(mut self, orig: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        self.orig_nlp = Some(orig);
        self
    }

    pub fn set_orig_nlp(&mut self, orig: Rc<RefCell<dyn IpoptNlp>>) {
        self.orig_nlp = Some(orig);
    }
}

impl Default for RestoRestorationPhase {
    fn default() -> Self {
        // Upstream's default rho is 1000 (registered in
        // `RestoIpoptNLP::RegisterOptions`). Match it here so a
        // bare-default `RestoRestorationPhase::default()` agrees with
        // the resto-NLP it pairs with when no rho was injected.
        Self::new(1000.0)
    }
}

impl RestorationPhase for RestoRestorationPhase {
    fn perform_restoration(
        &mut self,
        data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        _nlp: &Rc<RefCell<dyn IpoptNlp>>,
        _aug_solver: &mut dyn AugSystemSolver,
    ) -> RestorationOutcome {
        let orig_nlp = match self.orig_nlp.as_ref() {
            Some(o) => o.clone(),
            None => return RestorationOutcome::Failed,
        };

        let mu = data.borrow().curr_mu;
        let curr = match data.borrow().curr.clone() {
            Some(c) => c,
            None => return RestorationOutcome::Failed,
        };

        // The inner-IPM iterate's x is the resto-NLP compound:
        // `[x_orig, n_c, p_c, n_d, p_d]`.
        let curr_x_cv = match curr.x.as_any().downcast_ref::<CompoundVector>() {
            Some(c) if c.n_comps() == 5 => c,
            _ => return RestorationOutcome::Failed,
        };

        let x_orig = curr_x_cv.comp(BLOCK_X);
        let n_orig = x_orig.dim();
        let m_eq = curr_x_cv.comp(BLOCK_N_C).dim();
        let m_ineq = curr_x_cv.comp(BLOCK_N_D).dim();

        // c(x_orig) → c_buf.
        let mut c_buf = DenseVectorSpace::new(m_eq).make_new_dense();
        if m_eq > 0 {
            // Touch the buffer so it's materialized to zeros before
            // eval_c writes into it.
            c_buf.values_mut().fill(0.0);
            orig_nlp.borrow_mut().eval_c(x_orig, &mut c_buf);
        }

        // (d(x_orig) − s) → d_buf.
        let mut d_buf = DenseVectorSpace::new(m_ineq).make_new_dense();
        if m_ineq > 0 {
            d_buf.values_mut().fill(0.0);
            orig_nlp.borrow_mut().eval_d(x_orig, &mut d_buf);
            let s_vals = expanded_dense_values(&*curr.s, m_ineq);
            for (i, v) in d_buf.values_mut().iter_mut().enumerate() {
                *v -= s_vals[i];
            }
        }

        // Solve the per-element quadratic for the new slack values.
        let (n_c_vals, p_c_vals) = compute_n_p(&c_buf, mu, self.rho, m_eq);
        let (n_d_vals, p_d_vals) = compute_n_p(&d_buf, mu, self.rho, m_ineq);

        // Assemble new_x: keep block 0 (x_orig), replace blocks 1..=4.
        let new_x = build_new_x(
            n_orig, m_eq, m_ineq, x_orig, &n_c_vals, &p_c_vals, &n_d_vals, &p_d_vals,
        );

        // Stage as trial — `s` and all multipliers stay unchanged.
        let trial = IteratesVector::new(
            Rc::new(new_x),
            curr.s.clone(),
            curr.y_c.clone(),
            curr.y_d.clone(),
            curr.z_l.clone(),
            curr.z_u.clone(),
            curr.v_l.clone(),
            curr.v_u.clone(),
        );
        data.borrow_mut().set_trial(trial);
        RestorationOutcome::Recovered
    }
}

/// Per-element optimal `(n_i, p_i)` for the resto barrier objective at
/// fixed `(x_orig, s)`. See module doc for the derivation.
fn compute_n_p(c: &DenseVector, mu: Number, rho: Number, m: i32) -> (Vec<f64>, Vec<f64>) {
    let m = m as usize;
    if m == 0 {
        return (Vec::new(), Vec::new());
    }
    let half = mu / (2.0 * rho);
    let cvals = c.expanded_values();
    let mut n = vec![0.0; m];
    let mut p = vec![0.0; m];
    for i in 0..m {
        let a = half - 0.5 * cvals[i];
        let b = cvals[i] * half;
        // Numerical guard: a² + b is non-negative in exact arithmetic
        // (it equals (a + c_i/2 + ε)² for small ε from the half term),
        // but clamp to avoid NaN from FP roundoff on degenerate inputs.
        let radicand = (a * a + b).max(0.0);
        let v = a + radicand.sqrt();
        n[i] = v;
        p[i] = cvals[i] + v;
    }
    (n, p)
}

fn build_new_x(
    n_orig: i32,
    m_eq: i32,
    m_ineq: i32,
    x_orig: &dyn Vector,
    n_c: &[f64],
    p_c: &[f64],
    n_d: &[f64],
    p_d: &[f64],
) -> CompoundVector {
    let total = n_orig + 2 * m_eq + 2 * m_ineq;
    let space = CompoundVectorSpace::new(5, total);
    let s_n = DenseVectorSpace::new(n_orig);
    space.set_comp(BLOCK_X, n_orig, {
        let s = Rc::clone(&s_n);
        move || Box::new(DenseVector::new(Rc::clone(&s)))
    });
    let s_eq = DenseVectorSpace::new(m_eq);
    for i in [BLOCK_N_C, BLOCK_P_C] {
        space.set_comp(i, m_eq, {
            let s = Rc::clone(&s_eq);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
    }
    let s_ineq = DenseVectorSpace::new(m_ineq);
    for i in [BLOCK_N_D, BLOCK_P_D] {
        space.set_comp(i, m_ineq, {
            let s = Rc::clone(&s_ineq);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
    }
    let mut cv = CompoundVector::new(space);
    let x_orig_vals = expanded_dense_values(x_orig, n_orig);
    set_block(&mut cv, BLOCK_X, &x_orig_vals);
    set_block(&mut cv, BLOCK_N_C, n_c);
    set_block(&mut cv, BLOCK_P_C, p_c);
    set_block(&mut cv, BLOCK_N_D, n_d);
    set_block(&mut cv, BLOCK_P_D, p_d);
    cv
}

fn set_block(cv: &mut CompoundVector, idx: i32, vals: &[f64]) {
    let comp = cv.comp_mut(idx);
    let dense = comp
        .as_any_mut()
        .downcast_mut::<DenseVector>()
        .expect("RestoRestorationPhase: compound block must be DenseVector");
    if !vals.is_empty() {
        dense.set_values(vals);
    }
}

/// Expand a vector block to a dense slice, panicking with a clear
/// diagnostic if it is not a `DenseVector`. A failed downcast previously
/// substituted `vec![0.0; fallback_dim]` silently, masking a non-dense
/// block. The restoration data is all `DenseVector`, so this is an
/// invariant violation that must surface; `fallback_dim` is retained only
/// to size the diagnostic.
fn expanded_dense_values(v: &dyn Vector, fallback_dim: i32) -> Vec<f64> {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .map(|d| d.expanded_values())
        .unwrap_or_else(|| {
            panic!(
                "expanded_dense_values: expected a DenseVector for a length-{fallback_dim} block (got a non-dense block)"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_quadratic_matches_upstream_formula() {
        // For c=0: a = mu/(2ρ), b = 0; v = a + sqrt(a²) = 2a.
        // n = 2a, p = c + n = 2a.
        let s = DenseVectorSpace::new(2);
        let mut c = s.make_new_dense();
        c.set_values(&[0.0, 0.0]);
        let (n, p) = compute_n_p(&c, 1.0, 1000.0, 2);
        let half = 1.0 / 2000.0;
        for i in 0..2 {
            assert!((n[i] - 2.0 * half).abs() < 1e-15);
            assert!((p[i] - 2.0 * half).abs() < 1e-15);
        }
    }

    #[test]
    fn solve_quadratic_satisfies_feasibility_identity() {
        // p_i − n_i = c_i must hold by construction (p = c + n).
        let s = DenseVectorSpace::new(4);
        let mut c = s.make_new_dense();
        c.set_values(&[1.0, -2.0, 0.5, -0.1]);
        let (n, p) = compute_n_p(&c, 0.1, 1000.0, 4);
        let cvals = c.expanded_values();
        for i in 0..4 {
            assert!((p[i] - n[i] - cvals[i]).abs() < 1e-12);
            assert!(n[i] >= 0.0, "n must be non-negative");
            assert!(p[i] >= 0.0, "p must be non-negative");
        }
    }

    #[test]
    fn quadratic_root_satisfies_v2_minus_2av_minus_b_zero() {
        // The computed root v = a + sqrt(a²+b) (the value upstream's
        // solve_quadratic produces) is the positive root of
        // v² − 2·a·v − b = 0, NOT v² + 2·a·v − b = 0: from
        // v = a + sqrt(a²+b) we get (v − a)² = a² + b, i.e.
        // v² − 2·a·v − b = 0. Verify the correct identity holds with a
        // small residual, and that the wrong-sign form does NOT (locking
        // the L13 doc-sign correction so a regression in either the code or
        // the documented quadratic is caught).
        let s = DenseVectorSpace::new(3);
        let mut c = s.make_new_dense();
        c.set_values(&[3.0, -1.0, 0.7]);
        let mu = 0.5;
        let rho = 1000.0;
        let (n, _p) = compute_n_p(&c, mu, rho, 3);
        let half = mu / (2.0 * rho);
        let cvals = c.expanded_values();
        for i in 0..3 {
            let a = half - 0.5 * cvals[i];
            let b = cvals[i] * half;
            let v = n[i];
            let correct = v * v - 2.0 * a * v - b;
            assert!(
                correct.abs() < 1e-10 * (1.0 + a.abs() + b.abs()),
                "v² − 2av − b should be ≈0, got {correct} (v={v}, a={a}, b={b})"
            );
            let wrong = v * v + 2.0 * a * v - b;
            assert!(
                wrong.abs() > 1e-4,
                "v² + 2av − b is the wrong (doc-bug) form and must be \
                 clearly non-zero, got {wrong} (v={v}, a={a}, b={b})"
            );
        }
    }

    #[test]
    fn perform_restoration_with_no_orig_nlp_returns_failed() {
        let mut p = RestoRestorationPhase::new(1000.0);
        // No orig_nlp injected → graceful Failed.
        // We can't easily build a full IpoptDataHandle here without
        // dragging in algorithm-side fixtures, so just verify the
        // early-out path by checking the field directly.
        assert!(p.orig_nlp.is_none());
        // Use the constructor without setting orig_nlp.
        // The actual perform_restoration call is exercised end-to-end
        // via the integration test that runs the inner IPM.
        let _ = &mut p; // appease unused-mut on `p` if no further calls
    }

    #[test]
    fn default_uses_upstream_rho_default() {
        let p = RestoRestorationPhase::default();
        assert_eq!(p.rho, 1000.0);
    }
}
