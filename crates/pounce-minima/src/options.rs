//! Options for the global search: which strategy to run and its knobs.
//!
//! Moved here from `pounce-cli`'s `cli.rs` (gh#966), where they were
//! `MinimaMethod` / `MinimaArgs`; the CLI still parses argv into them and
//! re-exports them under those names.

/// Global-search strategy for `--minima`. Mirrors the six methods of
/// `pounce.find_minima` (`python/pounce/_minima.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Random / Sobol' box sampling (restart).
    Multistart,
    /// Multi-Level Single Linkage clustering (Rinnooy Kan & Timmer 1987).
    Mlsl,
    /// Metropolis chain over minima (Wales & Doye 1997).
    Basinhopping,
    /// Repulsive Gaussian bumps (filled-function; Ge 1990).
    Flooding,
    /// Softened `1/‖x−x*‖^p` poles (deflation; Farrell et al. 2015).
    Deflation,
    /// Equal-height tunnel between descents (Levy & Montalvo 1985).
    Tunneling,
}

impl Method {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "multistart" => Self::Multistart,
            "mlsl" => Self::Mlsl,
            "basinhopping" => Self::Basinhopping,
            "flooding" => Self::Flooding,
            "deflation" => Self::Deflation,
            "tunneling" => Self::Tunneling,
            other => {
                return Err(format!(
                    "unknown --minima method '{other}'; choose from \
                     multistart, mlsl, basinhopping, flooding, deflation, tunneling"
                ));
            }
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Multistart => "multistart",
            Self::Mlsl => "mlsl",
            Self::Basinhopping => "basinhopping",
            Self::Flooding => "flooding",
            Self::Deflation => "deflation",
            Self::Tunneling => "tunneling",
        }
    }
}

/// Parsed `--minima` configuration. Shared knobs have concrete defaults;
/// strategy-specific knobs are `Option`s resolved per-method in the driver
/// (so `"auto"` widths and curvature-based amplitudes match
/// `pounce.find_minima`). Field semantics mirror `_minima.py` exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct MinimaOptions {
    pub method: Method,
    /// Target: stop once this many distinct minima are found (default 10).
    pub n_minima: usize,
    /// Budget: hard cap on solver calls (default `8 * n_minima`).
    pub max_solves: Option<usize>,
    /// Give-up: stop after this many solves in a row that find nothing new.
    pub patience: usize,
    /// Two minima within this scaled distance are the same (default 1e-4).
    pub dedup: f64,
    /// Smallest Hessian eigenvalue tolerated by saddle rejection (1e-6).
    pub psd_tol: f64,
    /// Reject a converged solve whose unscaled dual infeasibility exceeds
    /// `kkt_tol · max(1, max|∇f(x0)|, dual_scale)` (gh#983; default 1e-4).
    /// `None` (`--kkt-tol none`) disables the check.
    pub kkt_tol: Option<f64>,
    /// Seed for the sampler / Sobol' scramble (default 0; reproducible).
    pub seed: u64,
    /// Use a scrambled Sobol' sequence for box sampling (default true).
    pub sobol: bool,
    // ---- strategy-specific knobs (None ⇒ per-method default) ----
    pub sigma: Option<f64>,
    pub sigma_frac: Option<f64>,
    pub amplitude: Option<f64>,
    pub amp_margin: Option<f64>,
    pub eta: Option<f64>,
    pub power: Option<f64>,
    pub soft: Option<f64>,
    pub length: Option<f64>,
    pub length_frac: Option<f64>,
    pub gamma: Option<f64>,
    pub samples_per_round: Option<usize>,
    pub step: Option<f64>,
    pub temperature: Option<f64>,
    pub restart_jitter: Option<f64>,
    // ---- library-only knobs (no CLI flag; `pounce.find_minima` passes them) ----
    /// Per-dimension flooding widths; takes precedence over `sigma`.
    pub sigma_per_dim: Option<Vec<f64>>,
    /// Per-dimension deflation / tunneling lengths; takes precedence over
    /// `length`.
    pub length_per_dim: Option<Vec<f64>>,
    /// MLSL sample cap (default `max_solves * samples_per_round`).
    pub max_samples: Option<usize>,
    /// Flooding: factor a basin's bump grows by on a failed escape (3).
    pub amp_bump: Option<f64>,
    /// Flooding: cap on that growth (1e3).
    pub amp_bump_cap: Option<f64>,
    /// Flooding: bump height when no Hessian gives a curvature-based one (2).
    pub amp_fallback: Option<f64>,
}

impl MinimaOptions {
    /// The solve budget: `max_solves`, or `8 * n_minima` when unset.
    pub fn max_solves_or_default(&self) -> usize {
        self.max_solves.unwrap_or(8 * self.n_minima)
    }
}

impl Default for MinimaOptions {
    fn default() -> Self {
        Self {
            // Matches `find_minima`'s default `method="deflation"`.
            method: Method::Deflation,
            n_minima: 10,
            max_solves: None,
            patience: 8,
            dedup: 1e-4,
            psd_tol: 1e-6,
            kkt_tol: Some(1e-4),
            seed: 0,
            sobol: true,
            sigma: None,
            sigma_frac: None,
            amplitude: None,
            amp_margin: None,
            eta: None,
            power: None,
            soft: None,
            length: None,
            length_frac: None,
            gamma: None,
            samples_per_round: None,
            step: None,
            temperature: None,
            restart_jitter: None,
            sigma_per_dim: None,
            length_per_dim: None,
            max_samples: None,
            amp_bump: None,
            amp_bump_cap: None,
            amp_fallback: None,
        }
    }
}
