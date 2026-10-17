//! Start-point sampling for `--minima`: scrambled Sobol' or uniform box
//! sampling when a finite box is known, else Gaussian jitter around `x0`.
//! Mirrors `_sample` / `_make_sobol` in `python/pounce/_minima.py`.
//!
//! Reproducibility: the PRNG is a seeded `ChaCha8Rng` and the Sobol' source
//! is `sobol_burley` (Owen-scrambled, matching scipy's
//! `qmc.Sobol(scramble=True)`), both keyed off `--seed`.

use pounce_common::types::Number;
use rand::distributions::{Distribution, Standard};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

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
                        // `sobol_burley` has 256 dimensions and panics past
                        // them; dimension `j` reuses `j mod 256` under a
                        // scramble seed of its own per block of 256, the
                        // crate's documented workaround. Block 0 keeps the
                        // run's seed, so every n <= 256 search draws exactly
                        // the starts it always did.
                        let (block, dim) = (j as u32 / NUM_DIMS, j as u32 % NUM_DIMS);
                        sobol_burley::sample(idx, dim, block_seed(self.sobol_seed, block)) as f64
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

/// Dimensions `sobol_burley` provides.
const NUM_DIMS: u32 = sobol_burley::NUM_DIMENSIONS;

/// The scramble seed for block `block` of [`NUM_DIMS`] dimensions: the run's
/// own for block 0, a well-mixed derivative of it (splitmix64's finalizer)
/// for the rest, so blocks are scrambled independently.
fn block_seed(seed: u32, block: u32) -> u32 {
    if block == 0 {
        return seed;
    }
    let mut z = ((u64::from(seed) << 32) | u64::from(block)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
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

#[cfg(test)]
mod tests {
    use super::{NUM_DIMS, Sampler, block_seed};

    /// Past 256 dimensions the draw stays in the box, and the extra blocks
    /// are not copies of the first (each has its own scramble seed).
    #[test]
    fn sobol_sampling_extends_past_256_dimensions() {
        let n = 3 * NUM_DIMS as usize + 7;
        let (lo, hi) = (vec![-1.0; n], vec![1.0; n]);
        let mut s = Sampler::new(5, true);
        for _ in 0..4 {
            let x = s.sample(&vec![0.0; n], &lo, &hi, true, 1.0);
            assert_eq!(x.len(), n);
            assert!(x.iter().all(|v| (-1.0..=1.0).contains(v)));
            let b = NUM_DIMS as usize;
            assert_ne!(x[..b], x[b..2 * b], "block 1 repeats block 0");
        }
        assert_eq!(block_seed(42, 0), 42);
        assert_ne!(block_seed(42, 1), block_seed(42, 2));
    }

    /// The first 256 dimensions draw what they always drew.
    #[test]
    fn the_first_256_dimensions_are_unchanged() {
        let n = NUM_DIMS as usize;
        let (lo, hi) = (vec![0.0; n], vec![1.0; n]);
        let mut s = Sampler::new(9, true);
        let x = s.sample(&vec![0.0; n], &lo, &hi, true, 1.0);
        let seed = (9u64 ^ (9u64 >> 32)) as u32;
        for (j, v) in x.iter().enumerate() {
            assert_eq!(*v, sobol_burley::sample(0, j as u32, seed) as f64);
        }
    }
}
