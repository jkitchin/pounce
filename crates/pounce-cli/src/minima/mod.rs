//! Multistart / find-minima driver for the `pounce` CLI (`--minima`).
//!
//! A pure-Rust port of `pounce.find_minima` (`python/pounce/_minima.py`):
//! drive the same local IPM solver in a loop, escaping already-found minima
//! by one of six strategies, and collect the distinct local minima into a
//! deduplicated archive. The strategies and their references:
//!
//! * `multistart` — random / Sobol' box sampling.
//! * `mlsl` — Multi-Level Single Linkage clustering (Rinnooy Kan & Timmer 1987).
//! * `basinhopping` — Metropolis chain over minima (Wales & Doye 1997).
//! * `flooding` — repulsive Gaussian bumps (filled-function; Ge 1990).
//! * `deflation` — softened `1/‖x−x*‖^p` poles (Farrell et al. 2015).
//! * `tunneling` — equal-height tunnel between descents (Levy & Montalvo 1985).
//!
//! The local solver is reused across starts on a single `IpoptApplication`
//! (no rebuild per start): each start wraps the base TNLP in a
//! [`SeededTnlp`] (and, for the repulsion strategies, a penalty wrapper).
//! Acceptance mirrors `_minima.py`: solve succeeded ∧ finite ∧ in-bounds ∧
//! (Hessian PSD within `psd_tol`) ∧ not already in the archive.

pub mod archive;
pub mod penalty_tnlp;
pub mod sampling;

use crate::cli::{Args, MinimaArgs, MinimaMethod, ProblemSource};
use crate::seeded_tnlp::SeededTnlp;
use crate::solve_report::{InputDescriptor, ReportBuilder, status_to_solve_result_num};
use archive::{Archive, scaled_distance};
use penalty_tnlp::{Kernel, PenaltyTnlp, TunnelTnlp};
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{BoundsInfo, IndexStyle, SparsityRequest, StartingPoint, TNLP};
use sampling::{Sampler, clip};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;

/// AMPL bound sentinel: a bound beyond this magnitude counts as ±∞.
const BOUND_INF: Number = 1e19;
/// Above this dimension a dense symmetric eigendecomposition (cyclic
/// Jacobi, O(n³)) is too slow for the per-acceptance saddle-rejection
/// check, so we skip it and accept (matching `find_minima`'s `hess=None`).
const PSD_MAX_N: usize = 256;

/// Why the search stopped (mirrors `MinimaResult.status`).
#[derive(Clone, Copy, Debug)]
enum Stop {
    TargetReached,
    Converged,
    BudgetExhausted,
}

impl Stop {
    fn as_str(self) -> &'static str {
        match self {
            Stop::TargetReached => "target_reached",
            Stop::Converged => "converged",
            Stop::BudgetExhausted => "budget_exhausted",
        }
    }
}

/// One local solve's outcome (the captured minimizer + whether it converged).
struct SolveOut {
    success: bool,
    x: Vec<Number>,
}

/// On-converged capture slot: the lifted full-length primal `x` plus the
/// base-problem constraint duals `lambda` of the most recent solve.
type SolveCapture = Rc<RefCell<Option<(Vec<Number>, Vec<Number>)>>>;

/// The find-minima driver. Holds the single application and base problem and
/// runs the chosen strategy until a [`Stop`].
struct Driver<'a> {
    app: &'a mut IpoptApplication,
    base: Rc<RefCell<dyn TNLP>>,
    /// Filled by the `on_converged` hook with the converged primal (full
    /// length) plus the base-problem constraint duals; cleared before each
    /// solve, taken after.
    capture: SolveCapture,
    cfg: &'a MinimaArgs,
    n: usize,
    m: usize,
    nnz_h: usize,
    /// 1 when the TNLP emits Fortran (1-based) triplet indices, else 0.
    index_offset: usize,
    x0: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    has_box: bool,
    /// Per-dimension scale `L` (box width, 1.0 for unbounded dims).
    l_scale: Vec<Number>,
    sampler: Sampler,
    archive: Archive,
    stagnant: usize,
    n_solves: usize,
    max_solves: usize,
    /// Sampled points drawn so far (only MLSL counts against this).
    n_samples: usize,
    /// Hard ceiling on sampled points for solve-gated strategies (MLSL),
    /// so `--max-solves` bounds wall-clock even when the clustering filter
    /// rejects every sample (pounce#103).
    max_samples: usize,
    psd_skipped_logged: bool,
}

impl<'a> Driver<'a> {
    // ---- shared local-solver ops -------------------------------------

    /// Run one solve of `solve_tnlp`, toggling the Hessian mode, and return
    /// the converged minimizer (captured via the `on_converged` hook).
    /// Returns `Err(Stop::BudgetExhausted)` once the solve budget is spent.
    fn run_solve(
        &mut self,
        solve_tnlp: Rc<RefCell<dyn TNLP>>,
        exact_hessian: bool,
    ) -> Result<SolveOut, Stop> {
        if self.n_solves >= self.max_solves {
            return Err(Stop::BudgetExhausted);
        }
        self.n_solves += 1;
        // Penalty (repulsion) solves go quasi-Newton; clean / polish solves
        // keep the exact Hessian. The IPM rereads this option each solve.
        let line = if exact_hessian {
            "hessian_approximation exact\n"
        } else {
            "hessian_approximation limited-memory\n"
        };
        let _ = self.app.options_mut().read_from_str(line, true);
        *self.capture.borrow_mut() = None;
        let status = self.app.optimize_tnlp(solve_tnlp);
        let success = matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        );
        match self.capture.borrow_mut().take() {
            // Only the primal is needed for acceptance; the duals are recovered
            // later by `recover_duals` (a clean base re-solve) so a point
            // accepted from an augmented penalty/tunnel solve still gets the
            // base problem's multipliers.
            Some((x, _lambda)) if success => Ok(SolveOut { success: true, x }),
            // A failed solve has no usable captured point; acceptance needs
            // success anyway, so the empty x is never read.
            _ => Ok(SolveOut {
                success: false,
                x: Vec::new(),
            }),
        }
    }

    /// Recover the base-problem constraint duals at an accepted minimum `x`.
    /// The accepting solve may have run on an augmented (penalty / tunnel)
    /// objective, whose multipliers are not the base problem's, so re-solve the
    /// clean base objective once from `x` — it is already optimal, so this
    /// converges immediately — and take the captured `lambda`. Budget-exempt:
    /// the point is already kept, so this does not consume a `--max-solves`
    /// slot. Falls back to zeros if the recovery solve does not converge or the
    /// nlp exposes no user-facing duals (length ≠ `m`).
    fn recover_duals(&mut self, x: &[Number]) -> Vec<Number> {
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(
            Rc::clone(&self.base),
            x.to_vec(),
        )));
        let _ = self
            .app
            .options_mut()
            .read_from_str("hessian_approximation exact\n", true);
        *self.capture.borrow_mut() = None;
        let status = self.app.optimize_tnlp(t);
        let ok = matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        );
        match self.capture.borrow_mut().take() {
            Some((_x, lambda)) if ok && lambda.len() == self.m => lambda,
            _ => vec![0.0; self.m],
        }
    }

    fn solve_seeded(&mut self, seed_x: &[Number], exact: bool) -> Result<SolveOut, Stop> {
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(
            Rc::clone(&self.base),
            seed_x.to_vec(),
        )));
        self.run_solve(t, exact)
    }

    fn solve_penalty(&mut self, seed_x: &[Number], kernel: Kernel) -> Result<SolveOut, Stop> {
        let pen: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PenaltyTnlp::new(
            Rc::clone(&self.base),
            kernel,
        )));
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(pen, seed_x.to_vec())));
        self.run_solve(t, false)
    }

    fn solve_tunnel(
        &mut self,
        seed_x: &[Number],
        f_ref: Number,
        pole: Kernel,
    ) -> Result<SolveOut, Stop> {
        let tun: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(TunnelTnlp::new(
            Rc::clone(&self.base),
            f_ref,
            pole,
        )));
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(tun, seed_x.to_vec())));
        self.run_solve(t, false)
    }

    /// Clean objective value at `x` (the un-augmented problem).
    fn clean_f(&mut self, x: &[Number]) -> Option<Number> {
        self.base.borrow_mut().eval_f(x, true)
    }

    /// Is `x` inside the box, allowing for the solver's bound relaxation?
    ///
    /// The IPM lets a converged primal sit slightly *outside* a bound — by up
    /// to `bound_relax_factor · max(1, |bound|)` (the Ipopt default factor is
    /// `1e-8`). On problems whose optimum binds a large-magnitude limit (e.g.
    /// ACOPF generator/flow limits in the hundreds), that legal slack exceeds
    /// any fixed absolute tolerance, so a purely absolute test wrongly rejects
    /// every minimum (pounce#101). Use a bound-magnitude-relative tolerance
    /// comfortably above the relaxation but far below any real basin spacing.
    fn in_bounds(&self, x: &[Number]) -> bool {
        x.iter()
            .zip(&self.x_l)
            .zip(&self.x_u)
            .all(|((&xi, &lo), &hi)| coord_in_bounds(xi, lo, hi))
    }

    /// Dense objective Hessian at `x` (row-major n×n), or `None` when no
    /// exact Hessian is available / the problem is too large.
    fn obj_hessian_dense(&mut self, x: &[Number]) -> Option<Vec<Number>> {
        if self.n > PSD_MAX_N || self.nnz_h == 0 {
            return None;
        }
        let nnz = self.nnz_h;
        let mut irow = vec![0 as Index; nnz];
        let mut jcol = vec![0 as Index; nnz];
        {
            let mut b = self.base.borrow_mut();
            if !b.eval_h(
                None,
                false,
                1.0,
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut irow,
                    jcol: &mut jcol,
                },
            ) {
                return None;
            }
        }
        let lam = vec![0.0; self.m];
        let mut vals = vec![0.0; nnz];
        {
            let mut b = self.base.borrow_mut();
            if !b.eval_h(
                Some(x),
                true,
                1.0,
                Some(&lam),
                true,
                SparsityRequest::Values { values: &mut vals },
            ) {
                return None;
            }
        }
        let n = self.n;
        let mut dense = vec![0.0; n * n];
        for k in 0..nnz {
            let i = irow[k] as usize - self.index_offset;
            let j = jcol[k] as usize - self.index_offset;
            dense[i * n + j] += vals[k];
            if i != j {
                dense[j * n + i] += vals[k];
            }
        }
        Some(dense)
    }

    /// Reject saddles/maxima via the clean Hessian's smallest eigenvalue
    /// (accept when no Hessian is available — matching `find_minima`).
    fn is_minimum(&mut self, x: &[Number]) -> bool {
        if self.n > PSD_MAX_N {
            if !self.psd_skipped_logged {
                eprintln!(
                    "pounce: --minima saddle-rejection (PSD) check skipped — n={} exceeds the \
                     dense-eigendecomposition cap ({PSD_MAX_N}); accepting converged points as minima.",
                    self.n
                );
                self.psd_skipped_logged = true;
            }
            return true;
        }
        let dense = match self.obj_hessian_dense(x) {
            Some(d) => d,
            None => return true,
        };
        let n = self.n;
        let mut w = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        if !pounce_sensitivity::symmetric_eigen(&dense, n, &mut w, &mut v) {
            return true;
        }
        let min_eig = w.iter().cloned().fold(f64::INFINITY, f64::min);
        min_eig >= -self.cfg.psd_tol
    }

    /// Per-dimension width vector from a knob spec, mirroring
    /// `_resolve_lengths`: a scalar ⇒ isotropic; `None` ("auto") ⇒
    /// `frac · L` when a box is known, else `fallback`.
    fn resolve_lengths(
        &self,
        spec: Option<f64>,
        frac_default: f64,
        frac_override: Option<f64>,
        fallback: f64,
    ) -> Vec<Number> {
        match spec {
            Some(s) => vec![s; self.n],
            None => {
                let frac = frac_override.unwrap_or(frac_default);
                if self.has_box {
                    self.l_scale.iter().map(|&l| frac * l).collect()
                } else {
                    vec![fallback; self.n]
                }
            }
        }
    }

    /// Curvature-based escape height for a flooding bump at `center`
    /// (`margin · μ_min` of `diag(σ)·H·diag(σ)`); `None` when no Hessian.
    fn auto_amplitude(&mut self, center: &[Number], sigma: &[Number], margin: f64) -> Option<f64> {
        let h = self.obj_hessian_dense(center)?;
        let n = self.n;
        let mut s_mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                s_mat[i * n + j] = sigma[i] * sigma[j] * h[i * n + j];
            }
        }
        let mut w = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        if !pounce_sensitivity::symmetric_eigen(&s_mat, n, &mut w, &mut v) {
            return None;
        }
        let mu_min = w.iter().cloned().fold(f64::INFINITY, f64::min);
        Some(margin * mu_min.max(1e-12))
    }

    /// Draw a fresh start from the box (Sobol'/uniform) or jitter around x0.
    fn sample(&mut self, jitter: f64) -> Vec<Number> {
        let x0 = self.x0.clone();
        let lo = self.x_l.clone();
        let hi = self.x_u.clone();
        self.sampler.sample(&x0, &lo, &hi, self.has_box, jitter)
    }

    // ---- acceptance --------------------------------------------------

    /// Consider a candidate for the archive. With `polish`, first re-solve
    /// the clean objective from the candidate (exact Hessian) — the
    /// repulsion strategies escape on the augmented objective, then polish
    /// back onto the true one. Returns whether it was accepted.
    fn consider(
        &mut self,
        mut x: Vec<Number>,
        mut success: bool,
        polish: bool,
    ) -> Result<bool, Stop> {
        if success && polish {
            let r = self.solve_seeded(&x, true)?;
            success = r.success;
            if success {
                x = r.x;
            }
        }
        if !success {
            return self.reject();
        }
        let fval = match self.clean_f(&x) {
            Some(f) => f,
            None => return self.reject(),
        };
        let finite = x.iter().all(|v| v.is_finite()) && fval.is_finite();
        let accepted =
            finite && self.in_bounds(&x) && self.is_minimum(&x) && !self.archive.is_known(&x);
        if accepted {
            // Recover the base-problem duals at the accepted point before
            // archiving (issue #196, related): the search may have accepted a
            // point from an augmented penalty/tunnel solve whose multipliers
            // are not the base problem's.
            let lambda = self.recover_duals(&x);
            self.archive.add(x, lambda, fval);
            self.stagnant = 0;
            if self.archive.len() >= self.cfg.n_minima {
                return Err(Stop::TargetReached);
            }
            Ok(true)
        } else {
            self.reject()
        }
    }

    fn reject(&mut self) -> Result<bool, Stop> {
        self.stagnant += 1;
        if self.stagnant >= self.cfg.patience {
            return Err(Stop::Converged);
        }
        Ok(false)
    }

    /// Count one drawn sample against the sampling budget. MLSL's expensive
    /// work is *sampling* (an O(N²) single-linkage scan over a growing pool),
    /// not solving, so on a problem where the clustering filter rejects
    /// almost every sample no solve ever fires and `max_solves` cannot bound
    /// the loop (pounce#103). The sample budget gives it a hard ceiling.
    fn note_sample(&mut self) -> Result<(), Stop> {
        if self.n_samples >= self.max_samples {
            return Err(Stop::BudgetExhausted);
        }
        self.n_samples += 1;
        Ok(())
    }

    // ---- strategy loops (each runs until a Stop) ---------------------

    fn run(&mut self) -> Stop {
        let res = match self.cfg.method {
            MinimaMethod::Multistart => self.run_multistart(),
            MinimaMethod::Mlsl => self.run_mlsl(),
            MinimaMethod::Basinhopping => self.run_basinhopping(),
            MinimaMethod::Flooding => self.run_flooding(),
            MinimaMethod::Deflation => self.run_deflation(),
            MinimaMethod::Tunneling => self.run_tunneling(),
        };
        // The loops only exit by propagating a `Stop`; `Ok` is unreachable.
        res.err().unwrap_or(Stop::BudgetExhausted)
    }

    fn run_multistart(&mut self) -> Result<(), Stop> {
        let jitter = self.cfg.restart_jitter.unwrap_or(1.0);
        let x0 = self.x0.clone();
        let r = self.solve_seeded(&x0, true)?;
        self.consider(r.x, r.success, false)?;
        loop {
            let s = self.sample(jitter);
            let r = self.solve_seeded(&s, true)?;
            self.consider(r.x, r.success, false)?;
        }
    }

    fn run_mlsl(&mut self) -> Result<(), Stop> {
        let batch = self.cfg.samples_per_round.unwrap_or(20);
        let gamma = self.cfg.gamma.unwrap_or(2.0);
        let jitter = self.cfg.restart_jitter.unwrap_or(1.0);
        let n = self.n;
        let diag = (n as f64).sqrt();
        let mut pool_x: Vec<Vec<Number>> = Vec::new();
        let mut pool_f: Vec<Number> = Vec::new();
        let x0 = self.x0.clone();
        let r = self.solve_seeded(&x0, true)?;
        self.consider(r.x, r.success, false)?;
        loop {
            // Grow the pool; each draw counts against the sample budget, so a
            // round that solves nothing still drives the loop to terminate.
            for _ in 0..batch {
                self.note_sample()?;
                let s = self.sample(jitter);
                let f = self.clean_f(&s).unwrap_or(f64::INFINITY);
                pool_x.push(s);
                pool_f.push(f);
            }
            let bign = pool_x.len();
            let ne = bign.max(2) as f64;
            let radius = gamma * diag * (ne.ln() / ne).powf(1.0 / n as f64);
            let mut order: Vec<usize> = (0..bign).collect();
            order.sort_by(|&a, &b| {
                pool_f[a]
                    .partial_cmp(&pool_f[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in order {
                let si = pool_x[i].clone();
                let fi = pool_f[i];
                // Single-linkage: skip if a *better* sample is within radius.
                let better_near = (0..bign).any(|j| {
                    j != i
                        && pool_f[j] < fi
                        && scaled_distance(&si, &pool_x[j], &self.l_scale) < radius
                });
                if better_near || self.archive.near_any(&si, radius) {
                    continue;
                }
                let r = self.solve_seeded(&si, true)?;
                self.consider(r.x, r.success, false)?;
            }
        }
    }

    fn run_basinhopping(&mut self) -> Result<(), Stop> {
        let step = self.cfg.step.unwrap_or(0.5);
        let temperature = self.cfg.temperature.unwrap_or(1.0);
        let x0 = self.x0.clone();
        let r = self.solve_seeded(&x0, true)?;
        let mut cur = if r.success { r.x.clone() } else { x0.clone() };
        let mut cur_f = self.clean_f(&cur).unwrap_or(f64::INFINITY);
        self.consider(r.x, r.success, false)?;
        loop {
            let mut trial = self.sampler.perturb(&cur, &[step]);
            clip(&mut trial, &self.x_l, &self.x_u, self.has_box);
            let r = self.solve_seeded(&trial, true)?;
            if !r.success {
                self.consider(r.x, false, false)?;
                continue;
            }
            let new_x = r.x.clone();
            self.consider(r.x, true, false)?;
            let new_f = self.clean_f(&new_x).unwrap_or(f64::INFINITY);
            let accept_uphill = self.sampler.uniform() < (-(new_f - cur_f) / temperature).exp();
            if new_f < cur_f || accept_uphill {
                cur = new_x;
                cur_f = new_f;
            }
        }
    }

    fn run_flooding(&mut self) -> Result<(), Stop> {
        let sigma = self.resolve_lengths(self.cfg.sigma, 0.1, self.cfg.sigma_frac, 0.5);
        let inv_sigma2: Vec<f64> = sigma.iter().map(|&s| 1.0 / (s * s)).collect();
        let amp_spec = self.cfg.amplitude;
        let margin = self.cfg.amp_margin.unwrap_or(2.0);
        let bump_factor = 3.0;
        let bump_cap = 1e3;
        let fallback_amp = 2.0;
        let jitter = self.cfg.restart_jitter.unwrap_or(0.5);
        let x0 = self.x0.clone();
        let mut base_amp: Vec<f64> = Vec::new();
        let mut mult: Vec<f64> = Vec::new();
        let mut start = x0.clone();
        let mut last_center: Option<usize> = None;
        let mut fails = 0usize;
        loop {
            let centers = self.archive.xs.clone();
            while base_amp.len() < centers.len() {
                let k = base_amp.len();
                let a = match amp_spec {
                    Some(a) => a,
                    None => self
                        .auto_amplitude(&centers[k], &sigma, margin)
                        .unwrap_or(fallback_amp),
                };
                base_amp.push(a);
                mult.push(1.0);
            }
            let eff: Vec<f64> = (0..centers.len()).map(|k| base_amp[k] * mult[k]).collect();
            let polish = !centers.is_empty();
            let solve_out = if centers.is_empty() {
                self.solve_seeded(&start, true)?
            } else {
                let kernel = Kernel::Gauss {
                    centers: centers.clone(),
                    amps: eff,
                    inv_sigma2: inv_sigma2.clone(),
                };
                self.solve_penalty(&start, kernel)?
            };
            let accepted = self.consider(solve_out.x, solve_out.success, polish)?;
            if accepted {
                if let Some(last) = self.archive.xs.last() {
                    start = last.clone();
                }
                last_center = Some(self.archive.xs.len() - 1);
                fails = 0;
            } else if let Some(lc) = last_center {
                if mult[lc] < bump_cap && fails < 8 {
                    // Under-flooded the basin we started from: bump and retry.
                    mult[lc] *= bump_factor;
                    let scale: Vec<f64> = sigma.iter().map(|&s| 0.05 * s).collect();
                    start = self.sampler.perturb(&centers[lc], &scale);
                    fails += 1;
                    continue;
                }
                start = self.sample(jitter);
                last_center = None;
                fails = 0;
            } else {
                start = self.sample(jitter);
                last_center = None;
                fails = 0;
            }
        }
    }

    fn run_deflation(&mut self) -> Result<(), Stop> {
        let eta = self.cfg.eta.unwrap_or(1.0);
        let power = self.cfg.power.unwrap_or(2.0);
        let soft = self.cfg.soft.unwrap_or(1e-3);
        let length = self.resolve_lengths(self.cfg.length, 0.1, self.cfg.length_frac, 0.5);
        let inv_len2: Vec<f64> = length.iter().map(|&l| 1.0 / (l * l)).collect();
        let jitter = self.cfg.restart_jitter.unwrap_or(0.5);
        let q = power / 2.0;
        let mut start = self.x0.clone();
        loop {
            let centers = self.archive.xs.clone();
            let polish = !centers.is_empty();
            // Step a little off the pole so the first gradient is finite.
            let mut s = start.clone();
            if !centers.is_empty() && self.archive.is_known(&s) {
                let scale: Vec<f64> = length.iter().map(|&l| 0.1 * l).collect();
                s = self.sampler.perturb(&s, &scale);
            }
            let solve_out = if centers.is_empty() {
                self.solve_seeded(&s, true)?
            } else {
                let kernel = Kernel::Pole {
                    centers: centers.clone(),
                    eta,
                    q,
                    soft,
                    inv_len2: inv_len2.clone(),
                };
                self.solve_penalty(&s, kernel)?
            };
            let accepted = self.consider(solve_out.x, solve_out.success, polish)?;
            if accepted {
                if let Some(last) = self.archive.xs.last() {
                    start = last.clone();
                }
            } else {
                start = self.sample(jitter);
            }
        }
    }

    fn run_tunneling(&mut self) -> Result<(), Stop> {
        let eta = self.cfg.eta.unwrap_or(1.0);
        let power = self.cfg.power.unwrap_or(2.0);
        let soft = self.cfg.soft.unwrap_or(1e-3);
        let length = self.resolve_lengths(self.cfg.length, 0.1, self.cfg.length_frac, 0.5);
        let inv_len2: Vec<f64> = length.iter().map(|&l| 1.0 / (l * l)).collect();
        let jitter = self.cfg.restart_jitter.unwrap_or(0.75);
        let q = power / 2.0;
        let x0 = self.x0.clone();
        // Seed: one clean descent.
        let r = self.solve_seeded(&x0, true)?;
        self.consider(r.x, r.success, false)?;
        loop {
            let centers = self.archive.xs.clone();
            // Tunnel at the height of the most-recent minimum, away from all
            // known minima — the classic monotone-descending tunnel.
            let f_ref = match self.archive.fs.last() {
                Some(&f) => f,
                None => self.clean_f(&x0).unwrap_or(0.0),
            };
            let anchor = self
                .archive
                .xs
                .last()
                .cloned()
                .unwrap_or_else(|| x0.clone());
            let jit = vec![jitter; self.n];
            let mut start = self.sampler.perturb(&anchor, &jit);
            clip(&mut start, &self.x_l, &self.x_u, self.has_box);
            let kernel = Kernel::Pole {
                centers: centers.clone(),
                eta,
                q,
                soft,
                inv_len2: inv_len2.clone(),
            };
            let r = self.solve_tunnel(&start, f_ref, kernel)?;
            self.consider(r.x, r.success, true)?;
        }
    }
}

/// A single found minimum (used for output / JSON).
struct Minimum {
    x: Vec<Number>,
    objective: Number,
    /// Base-problem constraint duals at this minimum (length `m`), recovered by
    /// a clean re-solve (issue #196, related). Zeros if unavailable.
    lambda: Vec<Number>,
}

/// Entry point: run the `--minima` search on `base` (the raw problem TNLP —
/// presolve / counting wrappers are intentionally bypassed so coordinates
/// match the original problem and the clean objective is evaluated directly).
/// Returns the process exit code.
pub fn run(
    app: &mut IpoptApplication,
    base: &Rc<RefCell<dyn TNLP>>,
    cfg: &MinimaArgs,
    args: &Args,
    sol_path: Option<&Path>,
) -> ExitCode {
    let info = match base.borrow_mut().get_nlp_info() {
        Some(i) => i,
        None => {
            eprintln!("pounce: --minima could not read problem dimensions");
            return ExitCode::from(2);
        }
    };
    let n = info.n as usize;
    let m = info.m as usize;
    let nnz_h = info.nnz_h_lag as usize;
    let index_offset = match info.index_style {
        IndexStyle::Fortran => 1,
        IndexStyle::C => 0,
    };

    // Bounds + starting point straight from the TNLP (works uniformly for
    // built-ins and `.nl` files).
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    base.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    });
    let mut x0 = vec![0.0; n];
    {
        let mut z_l = vec![0.0; n];
        let mut z_u = vec![0.0; n];
        let mut lambda = vec![0.0; m];
        base.borrow_mut().get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x0,
            init_z: false,
            z_l: &mut z_l,
            z_u: &mut z_u,
            init_lambda: false,
            lambda: &mut lambda,
        });
    }

    // Per-dimension scale and box availability (mirror `_scale_from_bounds`).
    let has_box = (0..n).all(|i| x_l[i] > -BOUND_INF && x_u[i] < BOUND_INF);
    let l_scale: Vec<Number> = (0..n)
        .map(|i| {
            let w = x_u[i] - x_l[i];
            if has_box && w > 0.0 { w } else { 1.0 }
        })
        .collect();

    // Capture the converged primal AND the base-problem constraint duals of
    // each solve via the on-converged hook. `lambda` uses the same
    // `finalize_solution_lambda` convention as the main NLP `.sol` path (c/d
    // split inversion + unscaling), so the `.sol` duals match a plain solve.
    // An nlp that does not expose user-facing duals returns an empty vec; that
    // (or any length mismatch) falls back to zeros where the duals are stored.
    let capture: SolveCapture = Rc::new(RefCell::new(None));
    {
        let cap = Rc::clone(&capture);
        app.set_on_converged(Box::new(move |data, _cq, nlp, _pd| {
            if let Some(curr) = data.borrow().curr.clone() {
                let nlp_ref = nlp.borrow();
                let x = nlp_ref.lift_x_to_full(&*curr.x);
                let lambda = nlp_ref.finalize_solution_lambda(&*curr.y_c, &*curr.y_d);
                *cap.borrow_mut() = Some((x, lambda));
            }
        }));
    }

    let max_solves = cfg.max_solves.unwrap_or(8 * cfg.n_minima);
    // Sample ceiling for solve-gated strategies (MLSL): one round of samples
    // per unit of solve budget. The patience-on-stall rule normally
    // terminates first; this guarantees `--max-solves` bounds wall-clock even
    // when the clustering filter rejects everything (pounce#103).
    let batch = cfg.samples_per_round.unwrap_or(20).max(1);
    let max_samples = max_solves.saturating_mul(batch);

    println!(
        "Searching for up to {} minima via `{}` (max {} solves, seed {})...",
        cfg.n_minima,
        cfg.method.as_str(),
        max_solves,
        cfg.seed
    );

    let mut driver = Driver {
        app,
        base: Rc::clone(base),
        capture,
        cfg,
        n,
        m,
        nnz_h,
        index_offset,
        x0,
        x_l: x_l.clone(),
        x_u: x_u.clone(),
        has_box,
        l_scale: l_scale.clone(),
        sampler: Sampler::new(cfg.seed, cfg.sobol),
        archive: Archive::new(cfg.dedup, l_scale.clone()),
        stagnant: 0,
        n_solves: 0,
        max_solves,
        n_samples: 0,
        max_samples,
        psd_skipped_logged: false,
    };

    let stop = driver.run();
    let n_solves = driver.n_solves;

    // Rank the found minima by objective (best first).
    let order = driver.archive.order_by_objective();
    let minima: Vec<Minimum> = order
        .iter()
        .map(|&i| Minimum {
            x: driver.archive.xs[i].clone(),
            objective: driver.archive.fs[i],
            lambda: driver.archive.ls[i].clone(),
        })
        .collect();
    let best_obj = order.first().map(|&i| driver.archive.fs[i]);

    print_table(&minima, &l_scale, stop, n_solves);

    // Write the per-minimum `.sol` files: best → <stub>.sol, the rest →
    // ranked siblings <stub>.minNNN.sol.
    if let Some(sp) = sol_path {
        write_sol_files(sp, &minima, m);
    }

    // JSON report: the standard single-solve report for the best minimum,
    // plus a backward-compatible `minima` section listing all of them.
    if let Some(json_path) = &args.json_output {
        write_json_report(json_path, args, cfg, stop, n_solves, &minima, n, m, &info);
    }

    if best_obj.is_some() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// Print a ranked console table of the distinct minima found.
fn print_table(minima: &[Minimum], l_scale: &[Number], stop: Stop, n_solves: usize) {
    println!();
    println!(
        "find-minima: {} distinct minim{} in {} solves ({})",
        minima.len(),
        if minima.len() == 1 { "um" } else { "a" },
        n_solves,
        stop.as_str()
    );
    if minima.is_empty() {
        println!("  (no accepted minima — try raising --max-solves or --patience)");
        return;
    }
    println!("  rank        objective     dist-to-best");
    let best = &minima[0].x;
    for (rank, mn) in minima.iter().enumerate() {
        let d = scaled_distance(&mn.x, best, l_scale);
        println!("  {rank:>4}   {:>16.8e}   {:>14.6e}", mn.objective, d);
    }
}

/// Write `.sol` files: best to `sol_path`, ranked siblings alongside.
fn write_sol_files(sol_path: &Path, minima: &[Minimum], m: usize) {
    let zeros = vec![0.0; m];
    for (rank, mn) in minima.iter().enumerate() {
        let path = if rank == 0 {
            sol_path.to_path_buf()
        } else {
            sibling_sol_path(sol_path, rank)
        };
        let message = format!(
            "POUNCE {} find-minima rank {rank}: Solve_Succeeded",
            env!("CARGO_PKG_VERSION")
        );
        // Real base-problem duals recovered per minimum (issue #196, related);
        // `recover_duals` guarantees length `m`, but guard defensively.
        let lambda = if mn.lambda.len() == m {
            &mn.lambda
        } else {
            &zeros
        };
        let payload = crate::nl_writer::SolutionFile {
            message: &message,
            x: &mn.x,
            mult_g: lambda,
            solve_result_num: status_to_solve_result_num(ApplicationReturnStatus::SolveSucceeded),
            suffixes: &[],
        };
        match crate::nl_writer::write_sol_file(&path, &payload) {
            Ok(_) => eprintln!("pounce: wrote {}", path.display()),
            Err(e) => eprintln!("pounce: failed to write {}: {e}", path.display()),
        }
    }
}

/// `<stub>.sol` → `<stub>.minNNN.sol` for rank ≥ 1.
fn sibling_sol_path(sol_path: &Path, rank: usize) -> PathBuf {
    let mut stub = sol_path.to_path_buf();
    stub.set_extension(""); // drop `.sol`
    let base = stub.to_string_lossy().into_owned();
    PathBuf::from(format!("{base}.min{rank:03}.sol"))
}

/// Build the JSON report (standard best-solution report + `minima` section)
/// and write it.
#[allow(clippy::too_many_arguments)]
fn write_json_report(
    json_path: &Path,
    args: &Args,
    cfg: &MinimaArgs,
    stop: Stop,
    n_solves: usize,
    minima: &[Minimum],
    n: usize,
    m: usize,
    info: &pounce_nlp::tnlp::NlpInfo,
) {
    let input = match &args.problem {
        ProblemSource::Builtin(name) => InputDescriptor::Builtin { name: name.clone() },
        ProblemSource::NlFile(p) => InputDescriptor::NlFile {
            path: p.clone(),
            size_bytes: std::fs::metadata(p).ok().map(|md| md.len()),
        },
    };
    let mut builder = ReportBuilder::new(args.json_detail, input);
    builder.problem.n_variables = n as Index;
    builder.problem.n_constraints = m as Index;
    builder.problem.n_objectives = 1;
    builder.problem.nnz_jac_g = Some(info.nnz_jac_g);
    builder.problem.nnz_h_lag = Some(info.nnz_h_lag);
    if let Some(best) = minima.first() {
        builder.solution.status = ApplicationReturnStatus::SolveSucceeded;
        builder.solution.solve_result_num =
            status_to_solve_result_num(ApplicationReturnStatus::SolveSucceeded);
        builder.solution.objective = best.objective;
        builder.solution.x = best.x.clone();
        // Real base-problem duals for the best minimum (issue #196, related);
        // `recover_duals` guarantees length `m`, guard defensively.
        builder.solution.lambda = if best.lambda.len() == m {
            best.lambda.clone()
        } else {
            vec![0.0; m]
        };
    }
    let report = builder.finish();

    // Inject the `minima` section without a schema change: serialize, then
    // splice it into the top-level object.
    let mut value = match serde_json::to_value(&report) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pounce: failed to serialize minima report: {e}");
            return;
        }
    };
    let minima_json: Vec<serde_json::Value> = minima
        .iter()
        .map(|mn| {
            serde_json::json!({
                "x": mn.x,
                "objective": mn.objective,
            })
        })
        .collect();
    let values: Vec<Number> = minima.iter().map(|mn| mn.objective).collect();
    if let serde_json::Value::Object(map) = &mut value {
        map.insert(
            "minima".to_string(),
            serde_json::json!({
                "method": cfg.method.as_str(),
                "status": stop.as_str(),
                "n_solves": n_solves,
                "n_minima": minima.len(),
                "minima": minima_json,
                "values": values,
            }),
        );
    }
    match serde_json::to_string_pretty(&value) {
        Ok(s) => match std::fs::write(json_path, s) {
            Ok(_) => eprintln!("pounce: wrote {}", json_path.display()),
            Err(e) => eprintln!(
                "pounce: failed to write JSON report to {}: {e}",
                json_path.display()
            ),
        },
        Err(e) => eprintln!("pounce: failed to render minima report: {e}"),
    }
}

/// Per-coordinate box test with a bound-magnitude-relative tolerance that
/// absorbs the IPM's bound relaxation (`bound_relax_factor·max(1,|bound|)`,
/// Ipopt default factor `1e-8`). A purely absolute tolerance rejects minima
/// that legally bind large-magnitude limits (pounce#101); the relative term
/// tracks the relaxation while staying far below any real basin spacing.
fn coord_in_bounds(xi: Number, lo: Number, hi: Number) -> bool {
    const ATOL: Number = 1e-9;
    const RTOL: Number = 1e-6;
    let tol_lo = ATOL + RTOL * lo.abs().max(1.0);
    let tol_hi = ATOL + RTOL * hi.abs().max(1.0);
    xi >= lo - tol_lo && xi <= hi + tol_hi
}

#[cfg(test)]
mod tests {
    use super::coord_in_bounds;

    #[test]
    fn accepts_interior_point() {
        assert!(coord_in_bounds(0.0, -1.0, 1.0));
        assert!(coord_in_bounds(250.0, 0.0, 500.0));
    }

    #[test]
    fn accepts_bound_relaxed_point_at_large_magnitude() {
        // A converged primal may sit ~bound_relax_factor·|bound| (≈5e-6 at a
        // 500-unit limit) past the bound; that point is a legal minimum.
        assert!(coord_in_bounds(500.000005, 0.0, 500.0));
        assert!(coord_in_bounds(-500.000005, -500.0, 0.0));
    }

    #[test]
    fn rejects_genuinely_outside_point() {
        assert!(!coord_in_bounds(500.1, 0.0, 500.0));
        assert!(!coord_in_bounds(-1.1, -1.0, 1.0));
    }
}
