//! The computed quantities second-order cones add to [`IpoptCalculatedQuantities`]
//! (workstream C, `dev-notes/research/nlp-soc-design.md`).
//!
//! A cone's rows are free `d` rows; their slacks `s_K` carry the cone, and the
//! cone dual `z_K` is its own variable (`IpoptData::soc_z`), tied to the rows'
//! multipliers only through the `s`-stationarity residual `−y_d − z` — the
//! structure Ipopt gives a bound slack and its `v`. Everything here is a no-op when
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

/// The cones' fraction-to-boundary ceiling (see [`IpoptCalculatedQuantities::soc_alpha_primal_max`]).
const SOC_TAU_MAX: Number = 0.99;

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

    /// Split a flat cone vector (layout order) into its cones.
    fn split(layout: &SocLayout, flat: &[Number]) -> Vec<Vec<Number>> {
        let mut off = 0;
        layout
            .blocks
            .iter()
            .map(|rows| {
                let b = flat[off..off + rows.len()].to_vec();
                off += rows.len();
                b
            })
            .collect()
    }

    /// The stored cone duals, flattened; `−y_d` on the cone rows of `iv` if
    /// none are stored yet (before initialization finishes).
    fn soc_z_or(&self, z: Option<Vec<Number>>, iv: &IteratesVector) -> Option<Vec<Number>> {
        let layout = self.soc_layout()?;
        Some(z.unwrap_or_else(|| {
            let y = dense(&*iv.y_d);
            layout.blocks.iter().flatten().map(|&r| -y[r]).collect()
        }))
    }

    fn soc_pairs_of(&self, iv: &IteratesVector, z: Option<Vec<Number>>) -> Option<SocPairs> {
        let layout = self.soc_layout()?;
        let z = self.soc_z_or(z, iv)?;
        let s = dense(&*iv.s);
        Some(SocPairs {
            s: layout
                .blocks
                .iter()
                .map(|rows| rows.iter().map(|&r| s[r]).collect())
                .collect(),
            z: Self::split(&layout, &z),
        })
    }

    /// `(s_K, z_K)` per cone at the current iterate.
    pub fn curr_soc_pairs(&self) -> Option<SocPairs> {
        let z = self.data.borrow().soc_z.clone();
        self.soc_pairs_of(&self.curr_iv(), z)
    }

    /// `(s_K, z_K)` per cone at the trial iterate (the current `z` if the
    /// trial staged none).
    pub fn trial_soc_pairs(&self) -> Option<SocPairs> {
        let z = {
            let d = self.data.borrow();
            d.soc_z_trial.clone().or_else(|| d.soc_z.clone())
        };
        self.soc_pairs_of(&self.trial_iv(), z)
    }

    /// The current cone duals, flattened (initialized from `−y_d` if unset).
    pub fn curr_soc_z(&self) -> Option<Vec<Number>> {
        let z = self.data.borrow().soc_z.clone();
        self.soc_z_or(z, &self.curr_iv())
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
        let Some(p) = self.soc_pairs_of(iv, None) else {
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

    /// Write the cone rows' `s`-stationarity residual `−y_d − z` into `v`
    /// (`grad_lag_s`, which holds `−y_d` there: the rows have no bound
    /// multiplier). `trial` reads the trial cone duals.
    pub(crate) fn set_soc_grad_lag_s(&self, trial: bool, v: &mut dyn Vector) {
        let Some(layout) = self.soc_layout() else {
            return;
        };
        let z = if trial {
            let d = self.data.borrow();
            d.soc_z_trial.clone().or_else(|| d.soc_z.clone())
        } else {
            self.data.borrow().soc_z.clone()
        };
        let iv = if trial {
            self.trial_iv()
        } else {
            self.curr_iv()
        };
        let Some(z) = self.soc_z_or(z, &iv) else {
            return;
        };
        let zs = Self::split(&layout, &z);
        write_rows(v, &layout, |k, block| {
            for (b, zi) in block.iter_mut().zip(&zs[k]) {
                *b -= zi;
            }
        });
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
        let Some(p) = self.soc_pairs_of(&iv, None) else {
            return;
        };
        let y = dense(&*iv.y_d);
        let mut k = 0;
        write_rows(rhs_s, &layout, |kk, block| {
            let mut g = vec![0.0; block.len()];
            soc_slack::barrier_gradient(&p.s[kk], mu, &mut g);
            for (j, (b, gi)) in block.iter_mut().zip(&g).enumerate() {
                // The cone's z cancels out of the reduced row: eliminating
                // dz = μ s⁻¹ − z − H ds from `−dy − dz = y_d + z` leaves
                // `H ds − dy = y_d + μ s⁻¹`, whatever z is.
                *b = -y[layout.blocks[kk][j]] + gi;
            }
            k += 1;
        });
    }

    /// Recover the cone duals' step from a Newton solve and store it
    /// (`IpoptData::soc_dz`): `dz = μ s⁻¹ − z − H ds − c`, with `c` the
    /// Mehrotra term when the corrector ran (`None` otherwise).
    pub fn store_soc_dz(&self, delta: &IteratesVector, mu: Number, c: Option<&[Vec<Number>]>) {
        let Some(dz) = self.soc_dz_of(delta, mu, c) else {
            return;
        };
        self.data.borrow_mut().soc_dz = Some(dz.into_iter().flatten().collect());
    }

    fn soc_dz_of(
        &self,
        delta: &IteratesVector,
        mu: Number,
        c: Option<&[Vec<Number>]>,
    ) -> Option<Vec<Vec<Number>>> {
        let layout = self.soc_layout()?;
        let p = self.curr_soc_pairs()?;
        let ds = dense(&*delta.s);
        let mut out = Vec::with_capacity(layout.blocks.len());
        for (k, rows) in layout.blocks.iter().enumerate() {
            let m = rows.len();
            let (s, z) = (&p.s[k], &p.z[k]);
            let h = soc_slack::nt_block(s, z)?;
            let mut sinv = vec![0.0; m];
            soc_slack::inverse(s, &mut sinv);
            let mut dz = vec![0.0; m];
            for i in 0..m {
                let hds: Number = (0..m).map(|j| h[i * m + j] * ds[rows[j]]).sum();
                dz[i] = mu * sinv[i] - z[i] - hds - c.map_or(0.0, |c| c[k][i]);
            }
            out.push(dz);
        }
        Some(out)
    }

    /// Stage the trial cone duals `z + α Δz` after the line search set the
    /// trial iterate (`set_trial` clears them).
    pub fn stage_soc_z_trial(&self, alpha_dual: Number) {
        let Some(z) = self.curr_soc_z() else {
            return;
        };
        let dz = self.data.borrow().soc_dz.clone();
        let Some(dz) = dz else {
            return;
        };
        let t: Vec<Number> = z.iter().zip(&dz).map(|(a, b)| a + alpha_dual * b).collect();
        self.data.borrow_mut().soc_z_trial = Some(t);
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

    /// The cones' dual cap: every `z_K + α Δz_K` inside its cone, `Δz` the
    /// step [`Self::store_soc_dz`] recovered. The cone duals step with the
    /// dual step size, as bound multipliers do.
    pub fn soc_alpha_dual_max(&self, delta: &IteratesVector, tau: Number) -> Number {
        self.soc_alpha_max(delta, tau, true)
    }

    fn soc_alpha_max(&self, delta: &IteratesVector, tau: Number, dual: bool) -> Number {
        let (Some(layout), Some(p)) = (self.soc_layout(), self.curr_soc_pairs()) else {
            return 1.0;
        };
        // Ipopt's `τ = max(τ_min, 1 − μ)` lets a bound slack close all but
        // `μ` of its distance per step; a cone cannot afford that. Its
        // boundary is `u₀ = ‖u_{1..}‖`, a difference of two numbers of the
        // cone's own magnitude, and at `τ = 1 − 1e-9` the step lands within
        // rounding of it: on CBLIB `flay02m` the dual of an active cone of
        // magnitude 51 came out with `det z = −2.4e-13` (exact arithmetic),
        // the NT block was undefined, and the solve ended
        // `Error_In_Step_Computation` one iteration from convergence. Conic
        // interior-point codes keep the fraction constant; so do the cones.
        let tau = tau.min(SOC_TAU_MAX);
        let ds = dense(&*delta.s);
        let dz = if dual {
            match self.data.borrow().soc_dz.clone() {
                Some(dz) => Some(Self::split(&layout, &dz)),
                None => return 1.0,
            }
        } else {
            None
        };
        let mut alpha: Number = 1.0;
        for (k, rows) in layout.blocks.iter().enumerate() {
            let a = match &dz {
                Some(dz) => soc_slack::max_step(&p.z[k], &dz[k], tau),
                None => {
                    let d: Vec<Number> = rows.iter().map(|&r| ds[r]).collect();
                    soc_slack::max_step(&p.s[k], &d, tau)
                }
            };
            alpha = alpha.min(a);
        }
        alpha
    }

    /// The cones' Mehrotra second-order terms at the affine (μ = 0) step
    /// `delta_aff` (`dz_a = −z − H ds_a`), and add them to a Newton `s`
    /// right-hand side (`rhs_s += c`, the solver's `−rhs` convention).
    /// `None` if a cone has left the interior (nothing is added then).
    pub fn add_soc_mehrotra(
        &self,
        delta_aff: &IteratesVector,
        rhs_s: &mut dyn Vector,
    ) -> Option<Vec<Vec<Number>>> {
        let layout = self.soc_layout()?;
        let p = self.curr_soc_pairs()?;
        let dz = self.soc_dz_of(delta_aff, 0.0, None)?;
        let ds = dense(&*delta_aff.s);
        let mut terms = Vec::with_capacity(layout.blocks.len());
        for (k, rows) in layout.blocks.iter().enumerate() {
            let dsk: Vec<Number> = rows.iter().map(|&r| ds[r]).collect();
            terms.push(soc_slack::mehrotra_term(&p.s[k], &p.z[k], &dsk, &dz[k])?);
        }
        write_rows(rhs_s, &layout, |k, block| {
            for (b, c) in block.iter_mut().zip(&terms[k]) {
                *b += c;
            }
        });
        Some(terms)
    }

    /// The cone duals with every one that has collapsed toward its boundary
    /// lifted back off it, or `None` if none has (or no cones). The cone's
    /// form of the lower half of Ipopt's `kappa_sigma` bound-multiplier rule
    /// (`z ≥ μ / (κ s)`): when `√(det s · det z) < μ / κ`, only `z₀` moves, by
    /// the least amount that restores `det s · det z = (μ / κ)²`.
    pub fn soc_lifted_z(&self, mu: Number, kappa: Number) -> Option<Vec<Number>> {
        let p = self.curr_soc_pairs()?;
        let floor = mu / kappa;
        let mut changed = false;
        let mut out = Vec::new();
        for (s, z) in p.s.iter().zip(&p.z) {
            let mut z = z.clone();
            let ds = soc_slack::det(s);
            let dz = soc_slack::det(&z);
            if ds > 0.0 && ((ds * dz.max(0.0)).sqrt() < floor || z[0] <= 0.0) {
                let tail2: Number = z[1..].iter().map(|v| v * v).sum();
                let z0 = (floor * floor / ds + tail2).sqrt();
                if z0 > z[0] {
                    z[0] = z0;
                    changed = true;
                }
            }
            out.extend(z);
        }
        changed.then_some(out)
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
