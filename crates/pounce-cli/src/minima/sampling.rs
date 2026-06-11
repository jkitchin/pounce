//! Start-point sampling for `--minima`: scrambled Sobol' or uniform box
//! sampling when a finite box is known, else Gaussian jitter around `x0`.
//! Mirrors `_sample` / `_make_sobol` in `python/pounce/_minima.py`.
//!
//! Reproducibility: the PRNG is a seeded `ChaCha8Rng` and the Sobol' source
//! is `sobol_burley` (Owen-scrambled, matching scipy's
//! `qmc.Sobol(scramble=True)`), both keyed off `--seed`.

use pounce_common::types::Number;
use rand::distributions::{Distribution, Standard};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Seeded sampler shared across a `--minima` run.
pub struct Sampler {
    rng: ChaCha8Rng,
    /// Sobol' scramble key (derived from the run seed).
    sobol_seed: u32,
    /// Next Sobol' point index (advances by one per drawn sample).
    sobol_index: u32,
    /// Whether to draw box samples from the Sobol' sequence.
    sobol: bool,
}

impl Sampler {
    pub fn new(seed: u64, sobol: bool) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            // Fold the 64-bit seed into the 32-bit Sobol' scramble key.
            sobol_seed: (seed ^ (seed >> 32)) as u32,
            sobol_index: 0,
            sobol,
        }
    }

    /// A uniform deviate in `[0, 1)` (avoids the `gen` keyword/method so the
    /// code stays edition-2024 clean).
    pub fn uniform(&mut self) -> f64 {
        Standard.sample(&mut self.rng)
    }

    /// A standard-normal deviate via Box–Muller (avoids a `rand_distr` dep).
    pub fn standard_normal(&mut self) -> f64 {
        // u1 in (0,1] so ln() is finite.
        let u1: f64 = 1.0 - self.uniform();
        let u2: f64 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Draw a fresh start point.
    ///
    /// * `has_box` true ⇒ sample uniformly (Sobol' or PRNG) in `[lo, hi]`.
    /// * otherwise ⇒ `x0 + jitter · N(0, I)` (the unbounded fallback).
    pub fn sample(
        &mut self,
        x0: &[Number],
        lo: &[Number],
        hi: &[Number],
        has_box: bool,
        jitter: f64,
    ) -> Vec<Number> {
        let n = x0.len();
        if has_box {
            let idx = self.sobol_index;
            self.sobol_index += 1;
            (0..n)
                .map(|j| {
                    let u = if self.sobol {
                        // One Owen-scrambled Sobol' coordinate per dimension.
                        sobol_burley::sample(idx, j as u32, self.sobol_seed) as f64
                    } else {
                        self.uniform()
                    };
                    lo[j] + (hi[j] - lo[j]) * u
                })
                .collect()
        } else {
            (0..n)
                .map(|j| x0[j] + jitter * self.standard_normal())
                .collect()
        }
    }

    /// `anchor + scale · N(0, I)` — a local perturbation (basinhopping step,
    /// flooding re-seed, tunnelling outward step). `scale` may be a scalar
    /// or per-dimension (length-1 ⇒ isotropic).
    pub fn perturb(&mut self, anchor: &[Number], scale: &[Number]) -> Vec<Number> {
        anchor
            .iter()
            .enumerate()
            .map(|(j, &a)| {
                let s = if scale.len() == 1 { scale[0] } else { scale[j] };
                a + s * self.standard_normal()
            })
            .collect()
    }
}

/// Clip `x` into `[lo, hi]` when a finite box is known (no-op otherwise).
pub fn clip(x: &mut [Number], lo: &[Number], hi: &[Number], has_box: bool) {
    if !has_box {
        return;
    }
    for j in 0..x.len() {
        x[j] = x[j].clamp(lo[j], hi[j]);
    }
}
