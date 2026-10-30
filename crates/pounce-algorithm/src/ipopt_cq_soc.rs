//! The computed quantities second-order cones add to [`IpoptCalculatedQuantities`]
//! (workstream C, `dev-notes/research/nlp-soc-design.md`).
//!
//! A cone's rows are free `d` rows; their slacks `s_K` carry the cone and the
//! cone dual is `z_K = −y_d` on those rows. Everything here is a no-op when
//! the NLP declares no cones ([`IpoptNlp::soc_layout`] is `None`), which is
//! every model today's code path solves.

use super::IpoptCalculatedQuantities;
use crate::iterates_vector::IteratesVector;
use crate::soc_slack;
use pounce_common::types::Number;
use pounce_linalg::{DenseVector, Vector};
use pounce_nlp::soc::SocLayout;
use std::rc::Rc;

/// Each cone's slack and dual at one iterate, head first.
#[derive(Debug, Clone)]
pub struct SocPairs {
    pub s: Vec<Vec<Number>>,
    pub z: Vec<Vec<Number>>,
}

fn dense(v: &dyn Vector) -> Vec<Number> {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .expect("second-order cones: the d-space vectors are dense")
        .expanded_values()
}

/// Write `f(block, values)` into `v`'s cone rows.
fn write_rows(v: &mut dyn Vector, layout: &SocLayout, mut f: impl FnMut(usize, &mut [Number])) {
    let d = v
        .as_any_mut()
        .downcast_mut::<DenseVector>()
        .expect("second-order cones: the d-space vectors are dense");
    let vals = d.values_mut();
    for (k, rows) in layout.blocks.iter().enumerate() {
        let mut block: Vec<Number> = rows.iter().map(|&r| vals[r]).collect();
        f(k, &mut block);
        for (&r, b) in rows.iter().zip(&block) {
            vals[r] = *b;
        }
    }
}

impl IpoptCalculatedQuantities {
    /// The declared cones, if any.
    pub fn soc_layout(&self) -> Option<Rc<SocLayout>> {
        self.nlp.borrow().soc_layout()
    }

    fn soc_pairs_of(&self, iv: &IteratesVector) -> Option<SocPairs> {
        let layout = self.soc_layout()?;
        let s = dense(&*iv.s);
        let y = dense(&*iv.y_d);
        let pick = |v: &[Number], sign: Number| -> Vec<Vec<Number>> {
            layout
                .blocks
                .iter()
                .map(|rows| rows.iter().map(|&r| sign * v[r]).collect())
                .collect()
        };
        Some(SocPairs {
            s: pick(&s, 1.0),
            z: pick(&y, -1.0),
        })
    }

    /// `(s_K, z_K = −y_d)` per cone at the current iterate.
    pub fn curr_soc_pairs(&self) -> Option<SocPairs> {
        self.soc_pairs_of(&self.curr_iv())
    }

    /// `(s_K, z_K)` per cone at the trial iterate.
    pub fn trial_soc_pairs(&self) -> Option<SocPairs> {
        self.soc_pairs_of(&self.trial_iv())
    }

    /// The Nesterov–Todd block `H` (`H s = z`, row-major) of every cone at the
    /// current iterate: the cone's `(s, s)` block of the augmented system, as
    /// `Σ_s = V/S` is a bounded slack's. `None` with no cones; an inner `None`
    /// for a cone whose `s` or `z` has left the interior, which the fraction-
    /// to-boundary rule exists to prevent.
    pub fn curr_soc_nt_blocks(&self) -> Option<Vec<Option<Vec<Number>>>> {
        let p = self.curr_soc_pairs()?;
        Some(
            p.s.iter()
                .zip(&p.z)
                .map(|(s, z)| soc_slack::nt_block(s, z))
                .collect(),
        )
    }

    /// The cones' `(s, s)` blocks for the augmented system at the current
    /// iterate. A cone that has left the interior gets a NaN block, so the
    /// factorization fails loudly instead of solving a system without it.
    pub fn curr_soc_kkt_blocks(&self) -> Option<crate::kkt::aug_system_solver::SocKktBlocks> {
        let layout = self.soc_layout()?;
        let blocks = self.curr_soc_nt_blocks()?;
        let h = layout
            .blocks
            .iter()
            .zip(blocks)
            .map(|(rows, h)| h.unwrap_or_else(|| vec![Number::NAN; rows.len() * rows.len()]))
            .collect();
        Some(crate::kkt::aug_system_solver::SocKktBlocks {
            rows: layout.blocks.clone(),
            h,
        })
    }

    /// `Σ_K −(μ/2) log det s_K` at the given pairs; `+∞` if any `s_K` is not
    /// interior.
    pub(crate) fn soc_barrier(pairs: &SocPairs, mu: Number) -> Number {
        pairs.s.iter().map(|s| soc_slack::barrier(s, mu)).sum()
    }

    /// Add the cones' barrier gradient `−μ s_K⁻¹` to `v` (a `d`-space vector)
    /// on their rows, read from iterate `iv`.
    pub(crate) fn add_soc_barrier_grad_s(
        &self,
        iv: &IteratesVector,
        mu: Number,
        v: &mut dyn Vector,
    ) {
        let Some(layout) = self.soc_layout() else {
            return;
        };
        let Some(p) = self.soc_pairs_of(iv) else {
            return;
        };
        write_rows(v, &layout, |k, block| {
            let mut g = vec![0.0; block.len()];
            soc_slack::barrier_gradient(&p.s[k], mu, &mut g);
            for (b, gi) in block.iter_mut().zip(&g) {
                *b += gi;
            }
        });
    }

    /// Zero `v`'s cone rows. `grad_lag_s` there is `−y_d − z ≡ 0` by the
    /// identity `z = −y_d`: a cone row's `s`-stationarity is carried by its
    /// complementarity, not by its dual infeasibility.
    pub(crate) fn zero_soc_rows(&self, v: &mut dyn Vector) {
        if let Some(layout) = self.soc_layout() {
            write_rows(v, &layout, |_, block| block.fill(0.0));
        }
    }

    /// The `s` right-hand side the Newton system needs on cone rows, written
    /// into `rhs_s` (which holds `grad_lag_s`, zero there): `−y_d − μ s⁻¹`,
    /// the gradient of the barrier Lagrangian. With the solver's `−rhs`
    /// convention the row reads `H ds − dy_d = y_d + μ s⁻¹` — Ipopt's
    /// eliminated slack row `Σ ds − dy_d = y_d + μ/s` with `z = −y_d`.
    pub fn set_soc_rhs_s(&self, mu: Number, rhs_s: &mut dyn Vector) {
        let Some(layout) = self.soc_layout() else {
            return;
        };
        let iv = self.curr_iv();
        let Some(p) = self.soc_pairs_of(&iv) else {
            return;
        };
        write_rows(rhs_s, &layout, |k, block| {
            let mut g = vec![0.0; block.len()];
            soc_slack::barrier_gradient(&p.s[k], mu, &mut g);
            for ((b, gi), zi) in block.iter_mut().zip(&g).zip(&p.z[k]) {
                // −y_d = z
                *b = zi + gi;
            }
        });
    }

    /// `(Σ_K ⟨s_K, z_K⟩, number of cones)`: the cones' share of the average
    /// complementarity, one count per cone (`⟨s, z⟩ = μ` on the central path).
    pub fn curr_soc_compl_sum(&self) -> (Number, usize) {
        match self.curr_soc_pairs() {
            None => (0.0, 0),
            Some(p) => (
                p.s.iter()
                    .zip(&p.z)
                    .map(|(s, z)| s.iter().zip(z).map(|(a, b)| a * b).sum::<Number>())
                    .sum(),
                p.s.len(),
            ),
        }
    }

    /// Smallest and largest per-cone `⟨s_K, z_K⟩` (`None` with no cones).
    pub fn curr_soc_compl_range(&self) -> Option<(Number, Number)> {
        let p = self.curr_soc_pairs()?;
        let vals: Vec<Number> =
            p.s.iter()
                .zip(&p.z)
                .map(|(s, z)| s.iter().zip(z).map(|(a, b)| a * b).sum())
                .collect();
        Some((
            vals.iter().copied().fold(Number::INFINITY, Number::min),
            vals.iter().copied().fold(Number::NEG_INFINITY, Number::max),
        ))
    }

    /// `max_K ‖s_K ∘ z_K − μ e‖∞`: the cones' complementarity error at `μ`
    /// (`μ = 0` for the unbarriered error). `0` with no cones.
    pub fn curr_soc_compl_error(&self, mu: Number) -> Number {
        match self.curr_soc_pairs() {
            None => 0.0,
            Some(p) => {
                p.s.iter()
                    .zip(&p.z)
                    .map(|(s, z)| soc_slack::complementarity_error(s, z, mu))
                    .fold(0.0, Number::max)
            }
        }
    }

    /// The cones' primal fraction-to-boundary cap on a step: the largest
    /// `α ≤ 1` (times `τ`) keeping every `s_K + α Δs_K` inside its cone.
    /// `1` without cones.
    pub fn soc_alpha_primal_max(&self, delta: &IteratesVector, tau: Number) -> Number {
        self.soc_alpha_max(delta, tau, false)
    }

    /// The cones' dual cap: every `z_K + α Δz_K = −(y_d + α Δy_d)_K` inside
    /// its cone. The cone rows of `y_d` are the cone duals and step with the
    /// dual step size, as a bound multiplier does — not with `alpha_y`:
    /// tying them to the primal step lets a dual direction that leaves the
    /// cone (driven by a large objective gradient through `x`-stationarity)
    /// throttle the primal step to nothing. Measured on Rosenbrock in an
    /// inactive cone: `alpha_pr` fell to `1e-16` with the dual cap binding.
    pub fn soc_alpha_dual_max(&self, delta: &IteratesVector, tau: Number) -> Number {
        self.soc_alpha_max(delta, tau, true)
    }

    fn soc_alpha_max(&self, delta: &IteratesVector, tau: Number, dual: bool) -> Number {
        let (Some(layout), Some(p)) = (self.soc_layout(), self.curr_soc_pairs()) else {
            return 1.0;
        };
        let step = if dual {
            dense(&*delta.y_d)
        } else {
            dense(&*delta.s)
        };
        let sign = if dual { -1.0 } else { 1.0 };
        let mut alpha: Number = 1.0;
        for (k, rows) in layout.blocks.iter().enumerate() {
            let d: Vec<Number> = rows.iter().map(|&r| sign * step[r]).collect();
            let v = if dual { &p.z[k] } else { &p.s[k] };
            alpha = alpha.min(soc_slack::max_step(v, &d, tau));
        }
        alpha
    }

    /// Every cone row, flattened (the `y_d` entries that step with the dual
    /// step size). Empty without cones.
    pub fn soc_rows_flat(&self) -> Vec<usize> {
        self.soc_layout()
            .map(|l| l.blocks.iter().flatten().copied().collect())
            .unwrap_or_default()
    }

    /// `δ_sᵀ H δ_s` summed over cones: the cones' share of the quadratic
    /// model's `δ_sᵀ Σ_s δ_s` term.
    pub(crate) fn soc_dhd(&self, delta_s: &dyn Vector) -> Number {
        let (Some(layout), Some(blocks)) = (self.soc_layout(), self.curr_soc_nt_blocks()) else {
            return 0.0;
        };
        let ds = dense(delta_s);
        let mut acc = 0.0;
        for (rows, h) in layout.blocks.iter().zip(&blocks) {
            let Some(h) = h else { continue };
            let m = rows.len();
            for i in 0..m {
                for j in 0..m {
                    acc += ds[rows[i]] * h[i * m + j] * ds[rows[j]];
                }
            }
        }
        acc
    }
}
