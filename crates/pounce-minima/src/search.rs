//! The multistart / find-minima search driver.
//!
//! Drives a local solver in a loop, escaping already-found minima by one of
//! six strategies, and collects the distinct local minima into a
//! deduplicated archive. Moved here from `pounce-cli` (gh#966); it is the
//! same algorithm as `pounce.find_minima` (`python/pounce/_minima.py`). The
//! strategies and their references:
//!
//! * `multistart` — random / Sobol' box sampling.
//! * `mlsl` — Multi-Level Single Linkage clustering (Rinnooy Kan & Timmer 1987).
//! * `basinhopping` — Metropolis chain over minima (Wales & Doye 1997).
//! * `flooding` — repulsive Gaussian bumps (filled-function; Ge 1990).
//! * `deflation` — softened `1/‖x−x*‖^p` poles (Farrell et al. 2015).
//! * `tunneling` — equal-height tunnel between descents (Levy & Montalvo 1985).
//!
//! The local solves go through a [`LocalSolve`]; each start wraps the base
//! TNLP in a [`SeededTnlp`] (and, for the repulsion strategies, a penalty
//! wrapper). Acceptance mirrors `_minima.py`: solve succeeded ∧ unscaled
//! stationarity within `kkt_tol` (gh#983) ∧ finite ∧ in-bounds ∧ (Hessian PSD
//! within `psd_tol`) ∧ not already in the archive.

use crate::archive::{Archive, Duals, scaled_distance};
use crate::local_solve::{HessianMode, LocalOutcome, LocalSolve};
use crate::options::{Method, MinimaOptions};
use crate::penalty_tnlp::{Kernel, PenaltyTnlp, TunnelTnlp};
use crate::sampling::{Sampler, clip};
use pounce_common::types::{Index, Number};
use pounce_nlp::seeded_tnlp::SeededTnlp;
use pounce_nlp::tnlp::{BoundsInfo, IndexStyle, SparsityRequest, StartingPoint, TNLP};
use std::cell::RefCell;
use std::ops::ControlFlow;
use std::rc::Rc;

/// AMPL bound sentinel: a bound beyond this magnitude counts as ±∞.
const BOUND_INF: Number = 1e19;
/// Above this dimension a dense symmetric eigendecomposition (cyclic
/// Jacobi, O(n³)) is too slow for the per-acceptance saddle-rejection
/// check, so we skip it and accept (matching `find_minima`'s `hess=None`).
const PSD_MAX_N: usize = 256;

/// Why the search stopped (mirrors `MinimaResult.status`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    TargetReached,
    Converged,
    BudgetExhausted,
    /// MLSL's sample pool hit its cap before the solve budget was spent
    /// (gh#989: this used to read `budget_exhausted`, naming the wrong
    /// budget).
    SampleCapReached,
    /// A [`Hooks::on_accept`] callback returned `ControlFlow::Break`.
    Aborted,
}

impl Stop {
    /// The status string `find_minima` and the CLI's JSON report use.
    pub fn as_str(self) -> &'static str {
        match self {
            Stop::TargetReached => "target_reached",
            Stop::Converged => "converged",
            Stop::BudgetExhausted => "budget_exhausted",
            Stop::SampleCapReached => "sample_cap_reached",
            Stop::Aborted => "aborted",
        }
    }
}

/// A failed launch whose landing point violates the base constraints by more
/// than this (absolute, unscaled) ended infeasible: Ipopt's default
/// `constr_viol_tol`. See `Driver::is_repeat_infeasible_landing` (gh#1042 R4c).
const INFEASIBLE_VIOL: Number = 1e-4;

/// Default MLSL critical-radius scale (gh#989). With the radius formula in
/// `run_mlsl`, `gamma = 2` gave `r ~ 1.1` of a `1.41` box diagonal in the
/// first rounds on a 2-D box, so every sample had a better one "nearby" and
/// almost no solve launched before the sample cap (six-hump camel: 2 solves).
/// `0.5` is the same order as the Rinnooy Kan-Timmer critical radius and
/// matches `pounce.find_minima`'s default.
const MLSL_GAMMA: f64 = 0.5;

/// For every pool point, whether a strictly better point lies within
/// `radius` in the per-dimension-scaled metric.
///
/// The naive scan is `O(N²)` per round (gh#989: ~15 s at `max_solves = 300`
/// in the Python twin). This sweeps the points sorted by their first scaled
/// coordinate and only compares pairs whose first coordinates are within
/// `radius` — the same answer, with the work bounded by the number of
/// near pairs instead of all pairs.
fn better_sample_within(xs: &[Vec<Number>], fs: &[Number], l: &[Number], radius: f64) -> Vec<bool> {
    let m = xs.len();
    let mut out = vec![false; m];
    if m < 2 || xs[0].is_empty() {
        return out;
    }
    let key = |i: usize| xs[i][0] / l[0];
    let mut by_key: Vec<usize> = (0..m).collect();
    by_key.sort_by(|&a, &b| {
        key(a)
            .partial_cmp(&key(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (pos, &i) in by_key.iter().enumerate() {
        let ki = key(i);
        let fi = fs[i];
        let mut hit = false;
        for &j in by_key[pos + 1..].iter() {
            if key(j) - ki >= radius {
                break;
            }
            if fs[j] < fi && scaled_distance(&xs[i], &xs[j], l) < radius {
                hit = true;
                break;
            }
        }
        if !hit {
            for &j in by_key[..pos].iter().rev() {
                if ki - key(j) >= radius {
                    break;
                }
                if fs[j] < fi && scaled_distance(&xs[i], &xs[j], l) < radius {
                    hit = true;
                    break;
                }
            }
        }
        out[i] = hit;
    }
    out
}

/// One local solve's outcome (the captured minimizer + whether it converged).
struct SolveOut {
    success: bool,
    /// The returned point; a candidate only when `success`.
    x: Vec<Number>,
}

/// Caller hooks into the search, all optional ([`find_minima_with`]).
#[derive(Default)]
pub struct Hooks<'a> {
    /// Called with `(x, f)` each time a minimum is archived. Returning
    /// `ControlFlow::Break(())` ends the search with [`Stop::Aborted`].
    #[allow(clippy::type_complexity)]
    pub on_accept: Option<&'a mut dyn FnMut(&[Number], Number) -> ControlFlow<()>>,
    /// The dedup metric `d(a, b)`, replacing the per-dimension-scaled
    /// Euclidean distance `‖(a−b)/L‖` — e.g. a periodic one. It decides
    /// whether a candidate is already known (within `dedup`) and, for MLSL,
    /// whether a sample is near a known minimum; MLSL's sample-to-sample
    /// clustering stays in the scaled metric.
    #[allow(clippy::type_complexity)]
    pub distance: Option<&'a mut dyn FnMut(&[Number], &[Number]) -> Number>,
    /// Record a [`TraceEntry`] for every candidate considered.
    pub trace: bool,
}

/// One candidate the search considered (`MinimaResult::trace`).
#[derive(Clone, Debug)]
pub struct TraceEntry {
    /// The candidate, after the polish solve if there was one; empty when
    /// the solve returned no point.
    pub x: Vec<Number>,
    /// The base objective at `x` (`NaN` when it could not be evaluated).
    pub f: Number,
    /// Whether its local solve converged (and passed `kkt_tol`).
    pub success: bool,
    pub accepted: bool,
    /// Minima archived before this candidate.
    pub n_found: usize,
}

/// The find-minima driver. Holds the local solver and base problem and runs
/// the chosen strategy until a [`Stop`].
struct Driver<'a> {
    solver: &'a mut dyn LocalSolve,
    hooks: Hooks<'a>,
    trace: Vec<TraceEntry>,
    base: Rc<RefCell<dyn TNLP>>,
    cfg: &'a MinimaOptions,
    n: usize,
    m: usize,
    nnz_h: usize,
    /// The base problem has a real Hessian: a nonzero structure *and* the
    /// solver says it is exact ([`LocalSolve::has_exact_hessian`]). A frontend
    /// with no user Hessian may still declare a structure (for the L-BFGS
    /// workspace) and fill it with zeros, which must not be read as
    /// curvature.
    has_hessian: bool,
    /// 1 when the TNLP emits Fortran (1-based) triplet indices, else 0.
    index_offset: usize,
    x0: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    /// The base problem's constraint bounds, for [`Self::constraint_violation`].
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    has_box: bool,
    /// Per-dimension scale `L` (box width, 1.0 for unbounded dims).
    l_scale: Vec<Number>,
    sampler: Sampler,
    archive: Archive,
    stagnant: usize,
    /// Landing points of failed launches that end infeasible (gh#1042 R4c):
    /// never archived, never reported, never a candidate. A later failed
    /// launch landing within `dedup` of one is a repeat hit of a known
    /// infeasible attractor and costs no `patience` strike; see
    /// [`Self::is_repeat_infeasible_landing`].
    infeasible_landings: Vec<Vec<Number>>,
    n_solves: usize,
    max_solves: usize,
    /// Sampled points drawn so far (only MLSL counts against this).
    n_samples: usize,
    /// Hard ceiling on sampled points for solve-gated strategies (MLSL),
    /// so `--max-solves` bounds wall-clock even when the clustering filter
    /// rejects every sample (pounce#103).
    max_samples: usize,
    psd_skipped_logged: bool,
    /// Messages for the caller (e.g. the PSD check being skipped).
    warnings: Vec<String>,
    /// The problem's gradient scale, `max |∇f(x0)|`, for the relative
    /// `kkt_tol` bound (gh#983). Set at setup from the base TNLP; if that
    /// evaluation fails, from the first solve's `start_obj_grad_max`.
    grad_ref: Option<Number>,
    /// Converged solves vetoed by `kkt_tol` (`MinimaResult.n_kkt_rejected`).
    n_kkt_rejected: usize,
}

impl<'a> Driver<'a> {
    // ---- shared local-solver ops -------------------------------------

    /// Run one solve of `solve_tnlp` in the given Hessian mode and return the
    /// converged minimizer. Returns `Err(Stop::BudgetExhausted)` once the
    /// solve budget is spent.
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
        // keep the exact Hessian.
        let mode = if exact_hessian {
            HessianMode::Exact
        } else {
            HessianMode::QuasiNewton
        };
        let out = self.solver.solve(solve_tnlp, mode);
        // Only the primal is needed for acceptance; the duals are recovered
        // later by `recover_duals` (a clean base re-solve) so a point accepted
        // from an augmented penalty/tunnel solve still gets the base problem's
        // multipliers. A failed solve's point is carried only for the trace;
        // acceptance needs success.
        let success = out.converged && !self.kkt_rejects(&out);
        Ok(SolveOut { success, x: out.x })
    }

    /// Largest violation of the base problem's constraint bounds at `x`,
    /// `max_j max(g_l_j − g_j(x), g_j(x) − g_u_j, 0)`, unscaled. `None` when
    /// `x` is not a full-length point or `eval_g` fails; `Some(0.0)` with no
    /// constraints.
    fn constraint_violation(&mut self, x: &[Number]) -> Option<Number> {
        if x.len() != self.n || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if self.m == 0 {
            return Some(0.0);
        }
        let mut g = vec![0.0; self.m];
        if !self.base.borrow_mut().eval_g(x, true, &mut g) {
            return None;
        }
        let mut worst: Number = 0.0;
        for ((&gj, &lo), &hi) in g.iter().zip(&self.g_l).zip(&self.g_u) {
            if !gj.is_finite() {
                return None;
            }
            if lo > -BOUND_INF {
                worst = worst.max(lo - gj);
            }
            if hi < BOUND_INF {
                worst = worst.max(gj - hi);
            }
        }
        Some(worst)
    }

    /// gh#983: veto a converged solve whose **unscaled** stationarity residual
    /// exceeds `kkt_tol · max(1, g0, dual_scale)`. `Solve_Succeeded` is a
    /// verdict in the solver's scaled space, and gradient scaling frozen at a
    /// huge-gradient start has been seen to certify points whose unscaled
    /// residual is ~1. The bound is relative so an objective whose gradients
    /// legitimately live at `1e12` is not rejected wholesale; see
    /// [`kkt_bound`]. Mirrors `_Context.solve` in `_minima.py`.
    fn kkt_rejects(&mut self, out: &LocalOutcome) -> bool {
        let Some(tol) = self.cfg.kkt_tol else {
            return false;
        };
        let q = &out.quality;
        if self.grad_ref.is_none() && q.start_obj_grad_max.is_finite() {
            self.grad_ref = Some(q.start_obj_grad_max);
        }
        let dual = q.unscaled_dual_inf;
        let bound = kkt_bound(tol, q.unscaled_dual_scale, self.grad_ref.unwrap_or(0.0));
        if dual.is_finite() && dual > bound {
            self.n_kkt_rejected += 1;
            true
        } else {
            false
        }
    }

    /// Recover the base-problem multipliers at an accepted minimum `x`: the
    /// constraint duals and the bound multipliers. The accepting solve may
    /// have run on an augmented (penalty / tunnel) objective, whose
    /// multipliers are not the base problem's, so re-solve the clean base
    /// objective once from `x` — it is already optimal, so this converges
    /// immediately — and take the captured `lambda`, `z_l` and `z_u`.
    /// Budget-exempt: the point is already kept, so this does not consume a
    /// `--max-solves` slot. Falls back to zeros if the recovery solve does not
    /// converge or the nlp exposes no user-facing duals (length ≠ `m` / `n`).
    ///
    /// With no constraints the re-solve is skipped unless `x` sits on a finite
    /// bound: it was not free. Started exactly at an interior minimum, where
    /// `max |∇f|` is ~1e-10, it reads to the gh#983 audit as a badly scaled
    /// objective, which re-solves with the objective multiplied by ~1e10 —
    /// cheap under Newton, but under L-BFGS (Python without `hess=`) both
    /// attempts ran to `max_iter`: measured on 6-D Rastrigin, 145 011
    /// objective evaluations and 11 s for 6 search solves (gh#966). An
    /// interior point of a bound-only problem has no nonzero multiplier to
    /// recover, so zeros are its answer. A point *on* a bound does (its
    /// reduced cost is the gradient component there, which is what makes it
    /// nonzero), and there the gradient is not small, so the audit does not
    /// misread it.
    fn recover_duals(&mut self, x: &[Number]) -> Duals {
        let n = x.len();
        let zeros = || Duals {
            lambda: vec![0.0; self.m],
            z_l: vec![0.0; n],
            z_u: vec![0.0; n],
        };
        if self.m == 0 && !self.on_a_bound(x) {
            return Duals {
                lambda: Vec::new(),
                ..zeros()
            };
        }
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(
            Rc::clone(&self.base),
            x.to_vec(),
        )));
        let out = self.solver.solve(t, HessianMode::Exact);
        // A recovery that failed leaves the bound multipliers *empty*, not
        // zero: the `.sol` writer skips an empty pair, while a zero pair is a
        // claim that no bound is active, and `pounce verify` then holds the
        // point to the exact (not the projected) stationarity test, which a
        // correct minimum on a bound fails.
        if !out.converged || out.lambda.len() != self.m {
            return Duals {
                z_l: Vec::new(),
                z_u: Vec::new(),
                ..zeros()
            };
        }
        let ok_z = out.z_l.len() == n && out.z_u.len() == n;
        Duals {
            lambda: out.lambda,
            z_l: if ok_z { out.z_l } else { Vec::new() },
            z_u: if ok_z { out.z_u } else { Vec::new() },
        }
    }

    /// Does any component of `x` sit on (within `1e-6·max(1, |b|)` of) a
    /// finite bound? Only the bound-only `recover_duals` gate reads it, so a
    /// loose test costs at most one budget-exempt re-solve.
    fn on_a_bound(&self, x: &[Number]) -> bool {
        let near = |v: Number, b: Number| (v - b).abs() <= 1e-6 * b.abs().max(1.0);
        x.iter()
            .zip(&self.x_l)
            .zip(&self.x_u)
            .any(|((&v, &lo), &hi)| {
                (lo > -BOUND_INF && near(v, lo)) || (hi < BOUND_INF && near(v, hi))
            })
    }

    fn solve_seeded(&mut self, seed_x: &[Number], exact: bool) -> Result<SolveOut, Stop> {
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(
            Rc::clone(&self.base),
            seed_x.to_vec(),
        )));
        self.run_solve(t, exact)
    }

    /// One flooding / deflation solve on the augmented objective.
    ///
    /// On an unconstrained or bound-only problem with a Hessian, and `n` small
    /// enough for a dense block ([`PSD_MAX_N`], the saddle check's cap), it
    /// runs on the exact augmented Hessian; otherwise quasi-Newton. Measured
    /// over seeds 0–9 (`n_minima` 6, 30 solves, gh#966), runs that found all
    /// six minima, L-BFGS → exact: camel deflation 1 → 10 (255 → 232 solves
    /// in total), camel flooding 7 → 9, wavy unchanged at 10/10 both ways.
    /// With constraints it went the other way: `circle_parabola_t0p5`
    /// deflation found both its minima in 8/10 seeds under L-BFGS and 3/10
    /// exact (`haverly_multistart_t3_brf0` unchanged at 10/10), so a
    /// constrained problem keeps the quasi-Newton solve.
    fn solve_penalty(&mut self, seed_x: &[Number], kernel: Kernel) -> Result<SolveOut, Stop> {
        let base = Rc::clone(&self.base);
        let exact = if self.m == 0 && self.has_hessian && self.n <= PSD_MAX_N {
            PenaltyTnlp::with_exact_hessian(Rc::clone(&base), kernel.clone())
        } else {
            None
        };
        let use_exact = exact.is_some();
        let pen: Rc<RefCell<dyn TNLP>> = match exact {
            Some(p) => Rc::new(RefCell::new(p)),
            None => Rc::new(RefCell::new(PenaltyTnlp::new(base, kernel))),
        };
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SeededTnlp::new(pen, seed_x.to_vec())));
        self.run_solve(t, use_exact)
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
        if self.n > PSD_MAX_N || !self.has_hessian {
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
                self.warnings.push(format!(
                    "saddle-rejection (PSD) check skipped — n={} exceeds the \
                     dense-eigendecomposition cap ({PSD_MAX_N}); accepting converged points as minima.",
                    self.n
                ));
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
        if !pounce_linalg::symmetric_eigen(&dense, n, &mut w, &mut v) {
            return true;
        }
        let min_eig = w.iter().cloned().fold(f64::INFINITY, f64::min);
        min_eig >= -self.cfg.psd_tol
    }

    /// Per-dimension width vector from a knob spec; see [`resolve_lengths`].
    fn resolve_lengths(
        &self,
        per_dim: Option<&Vec<f64>>,
        spec: Option<f64>,
        frac_default: f64,
        frac_override: Option<f64>,
        fallback: f64,
    ) -> Vec<Number> {
        let frac = frac_override.unwrap_or(frac_default);
        resolve_lengths(per_dim, spec, frac, fallback, &self.l_scale, self.has_box)
    }

    /// Curvature-based escape height for a flooding bump at `center`; see
    /// [`auto_amplitude`]. `None` when no Hessian.
    fn auto_amplitude(&mut self, center: &[Number], sigma: &[Number], margin: f64) -> Option<f64> {
        let h = self.obj_hessian_dense(center)?;
        auto_amplitude(&h, sigma, margin)
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
            x = r.x;
        }
        let fval = if success || (self.hooks.trace && !x.is_empty()) {
            self.clean_f(&x)
        } else {
            None
        };
        let accepted = success
            && fval.is_some_and(|f| f.is_finite())
            && x.iter().all(|v| v.is_finite())
            && self.in_bounds(&x)
            && self.is_minimum(&x)
            && !self.is_known(&x);
        if self.hooks.trace {
            self.trace.push(TraceEntry {
                x: x.clone(),
                f: fval.unwrap_or(Number::NAN),
                success,
                accepted,
                n_found: self.archive.len(),
            });
        }
        if !accepted {
            if !success && self.is_repeat_infeasible_landing(&x) {
                return Ok(false);
            }
            return self.reject();
        }
        let fval = fval.unwrap_or(Number::NAN);
        // Recover the base-problem duals at the accepted point before
        // archiving (issue #196, related): the search may have accepted a
        // point from an augmented penalty/tunnel solve whose multipliers are
        // not the base problem's.
        let duals = self.recover_duals(&x);
        self.archive.add(x, duals, fval);
        self.stagnant = 0;
        if let Some(cb) = self.hooks.on_accept.as_mut() {
            let last = self.archive.xs.last().map(Vec::as_slice).unwrap_or(&[]);
            if cb(last, fval).is_break() {
                return Err(Stop::Aborted);
            }
        }
        if self.archive.len() >= self.cfg.n_minima {
            return Err(Stop::TargetReached);
        }
        Ok(true)
    }

    /// Is `x` within `dedup` of an archived minimum, in the caller's metric
    /// when one is given?
    fn is_known(&mut self, x: &[Number]) -> bool {
        let dedup = self.cfg.dedup;
        self.near_any(x, dedup)
    }

    /// Is `x` within `radius` of an archived minimum (MLSL clustering)?
    fn near_any(&mut self, x: &[Number], radius: Number) -> bool {
        match self.hooks.distance.as_mut() {
            Some(d) => self.archive.xs.iter().any(|m| d(x, m) <= radius),
            None => self.archive.near_any(x, radius),
        }
    }

    /// gh#1042 R4c: is this failed launch a *repeat* hit of an infeasible
    /// point the search has already seen? Records the point when it is a
    /// first hit.
    ///
    /// `patience` counts consecutive draws that found nothing new, as
    /// evidence the landscape is exhausted. A launch that lands on an
    /// infeasible attractor the search has already seen is not such
    /// evidence — it is one more sample of that attractor's basin. Measured
    /// on `circle_parabola_t0p5` (MLSL, seeds 0–9, gamma 0.5): every failed
    /// launch on every seed ends `Converged to a point of local
    /// infeasibility` at one point, (0, −1), and on seeds 1, 2, 4 six of the
    /// eight strikes that stopped the search were repeats of it, with the
    /// second minimum reachable at solve 10–17 of 30.
    ///
    /// "Infeasible" is read from the point, not the solver's status: the
    /// base constraints are violated by more than [`INFEASIBLE_VIOL`] (Ipopt's
    /// default `constr_viol_tol`, the solver's own line between feasible and
    /// not). A failed launch that ends feasible (`MaxIter`, `ErrorInStep` at
    /// a feasible iterate) keeps its full strike. Over the R4c classifier
    /// check (the 10-seed grid at patience 8 and 1000 plus the
    /// `sweep-minima.sh` runs, 1848 solves) this flags exactly the 229
    /// launches whose exit was local infeasibility (violation 1.5) and none
    /// of the 19 feasible failures (worst violation 5.3e-5, a `MaxIter`).
    ///
    /// MLSL only. Multistart is the other independent-draw strategy, and was
    /// measured with the rule on: `circle_parabola_t0p5` seeds 0–9 found
    /// both minima on 7/10 seeds either way while mean solves rose 10.8 →
    /// 14.5, so it buys multistart nothing (plausibly because MLSL's
    /// clustering filter keeps samples near an archived minimum from
    /// launching, leaving its launches dominated by the infeasible basin;
    /// not measured further). The repulsion strategies' next start depends
    /// on the last result, and their failed solves are on an augmented
    /// problem.
    ///
    /// The first hit of each distinct infeasible point is still a full
    /// strike, so launches that go infeasible in many different places still
    /// exhaust `patience`; and until a minimum is archived every hit counts,
    /// so a model infeasible from everywhere the sampler reaches still stops
    /// at `patience`. `max_solves` and the sample cap bound the loop either
    /// way.
    fn is_repeat_infeasible_landing(&mut self, x: &[Number]) -> bool {
        if self.cfg.method != Method::Mlsl {
            return false;
        }
        if !self
            .constraint_violation(x)
            .is_some_and(|v| v > INFEASIBLE_VIOL)
        {
            return false;
        }
        let dedup = self.cfg.dedup;
        let l = &self.l_scale;
        let seen = match self.hooks.distance.as_mut() {
            Some(d) => self.infeasible_landings.iter().any(|p| d(x, p) <= dedup),
            None => self
                .infeasible_landings
                .iter()
                .any(|p| scaled_distance(x, p, l) <= dedup),
        };
        if !seen {
            self.infeasible_landings.push(x.to_vec());
            return false;
        }
        !self.archive.is_empty()
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
            return Err(Stop::SampleCapReached);
        }
        self.n_samples += 1;
        Ok(())
    }

    // ---- strategy loops (each runs until a Stop) ---------------------

    fn run(&mut self) -> Stop {
        let res = match self.cfg.method {
            Method::Multistart => self.run_multistart(),
            Method::Mlsl => self.run_mlsl(),
            Method::Basinhopping => self.run_basinhopping(),
            Method::Flooding => self.run_flooding(),
            Method::Deflation => self.run_deflation(),
            Method::Tunneling => self.run_tunneling(),
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
        let gamma = self.cfg.gamma.unwrap_or(MLSL_GAMMA);
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
            let better_near = better_sample_within(&pool_x, &pool_f, &self.l_scale, radius);
            let mut order: Vec<usize> = (0..bign).collect();
            order.sort_by(|&a, &b| {
                pool_f[a]
                    .partial_cmp(&pool_f[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for i in order {
                // Single-linkage: skip if a *better* sample is within radius.
                if better_near[i] || self.near_any(&pool_x[i], radius) {
                    continue;
                }
                let si = pool_x[i].clone();
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
        let sigma = self.resolve_lengths(
            self.cfg.sigma_per_dim.as_ref(),
            self.cfg.sigma,
            0.1,
            self.cfg.sigma_frac,
            0.5,
        );
        let inv_sigma2: Vec<f64> = sigma.iter().map(|&s| 1.0 / (s * s)).collect();
        let amp_spec = self.cfg.amplitude;
        let margin = self.cfg.amp_margin.unwrap_or(2.0);
        let bump_factor = self.cfg.amp_bump.unwrap_or(3.0);
        let bump_cap = self.cfg.amp_bump_cap.unwrap_or(1e3);
        let fallback_amp = self.cfg.amp_fallback.unwrap_or(2.0);
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
        let length = self.resolve_lengths(
            self.cfg.length_per_dim.as_ref(),
            self.cfg.length,
            0.1,
            self.cfg.length_frac,
            0.5,
        );
        let inv_len2: Vec<f64> = length.iter().map(|&l| 1.0 / (l * l)).collect();
        let jitter = self.cfg.restart_jitter.unwrap_or(0.5);
        let q = power / 2.0;
        let mut start = self.x0.clone();
        loop {
            let centers = self.archive.xs.clone();
            let polish = !centers.is_empty();
            // Step a little off the pole so the first gradient is finite.
            let mut s = start.clone();
            if !centers.is_empty() && self.is_known(&s) {
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
        let length = self.resolve_lengths(
            self.cfg.length_per_dim.as_ref(),
            self.cfg.length,
            0.1,
            self.cfg.length_frac,
            0.5,
        );
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

/// A distinct local minimum the search found.
#[derive(Clone, Debug)]
pub struct Minimum {
    /// The minimizer, full length, in the problem's own coordinates.
    pub x: Vec<Number>,
    /// The base (un-augmented) objective at `x`.
    pub objective: Number,
    /// Base-problem constraint duals at this minimum (length `m`), recovered
    /// by a clean re-solve from `x` (issue #196, related): the accepting
    /// solve may have run on an augmented penalty / tunnel objective, whose
    /// multipliers are not the base problem's. Zeros if unavailable.
    pub lambda: Vec<Number>,
    /// Base-problem lower / upper bound multipliers at this minimum (length
    /// `n`), from the same re-solve as `lambda`: `≥ 0` at an active bound,
    /// in the frame of the problem the engine solved (for a `maximize` `.nl`,
    /// `min -f`), so a writer applies the `.sol` signs
    /// `ipopt_zL_out = σ·z_l`, `ipopt_zU_out = −σ·z_u` exactly as a plain
    /// solve does. Zeros at an interior minimum of a problem without
    /// constraints (see `recover_duals`); empty if the recovery re-solve
    /// failed, so nothing is written rather than a false "no bound active".
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
}

/// What [`find_minima`] returns.
#[derive(Clone, Debug)]
pub struct MinimaResult {
    /// The distinct minima, best (lowest objective) first.
    pub minima: Vec<Minimum>,
    /// Why the search stopped.
    pub stop: Stop,
    /// Local solves spent against `max_solves` (dual-recovery re-solves are
    /// not counted).
    pub n_solves: usize,
    /// Converged solves vetoed by `kkt_tol` (gh#983).
    pub n_kkt_rejected: usize,
    /// The per-dimension length scale the dedup metric uses (box width per
    /// variable, `1` for an unbounded one), for reporting distances in the
    /// same units.
    pub scale: Vec<Number>,
    /// Messages for the caller, e.g. that the saddle-rejection check was
    /// skipped because the problem is too large for it.
    pub warnings: Vec<String>,
    /// Every candidate considered, when [`Hooks::trace`] was set.
    pub trace: Vec<TraceEntry>,
}

/// [`find_minima`] could not start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MinimaError {
    /// `get_nlp_info` returned `None`.
    NoProblemInfo,
}

impl std::fmt::Display for MinimaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProblemInfo => f.write_str("could not read problem dimensions"),
        }
    }
}

impl std::error::Error for MinimaError {}

/// Run the global search on `base`, solving each start with `solver`.
///
/// `base` is the raw problem: the CLI deliberately bypasses presolve and its
/// counting wrappers, so coordinates match the original problem and the clean
/// objective is evaluated directly. Its bounds and starting point come
/// straight from the TNLP; the search draws further starts from the box (or
/// jitters around `x0` where there is none).
pub fn find_minima(
    base: &Rc<RefCell<dyn TNLP>>,
    solver: &mut dyn LocalSolve,
    cfg: &MinimaOptions,
) -> Result<MinimaResult, MinimaError> {
    find_minima_with(base, solver, cfg, Hooks::default())
}

/// [`find_minima`] with caller [`Hooks`]: an accept callback, a custom dedup
/// metric, and a trace of every candidate.
pub fn find_minima_with<'a>(
    base: &Rc<RefCell<dyn TNLP>>,
    solver: &'a mut dyn LocalSolve,
    cfg: &'a MinimaOptions,
    hooks: Hooks<'a>,
) -> Result<MinimaResult, MinimaError> {
    let info = base
        .borrow_mut()
        .get_nlp_info()
        .ok_or(MinimaError::NoProblemInfo)?;
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

    // The gradient scale `max |∇f(x0)|` the `kkt_tol` bound is relative to
    // (gh#983), as `find_minima` takes it from `jac(x0)`.
    let grad_ref = cfg.kkt_tol.and_then(|_| {
        let mut g = vec![0.0; n];
        let ok = base.borrow_mut().eval_grad_f(&x0, true, &mut g);
        let g0 = g.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        (ok && g0.is_finite()).then_some(g0)
    });

    // Per-dimension scale and box availability (mirror `_scale_from_bounds`).
    let has_box = (0..n).all(|i| x_l[i] > -BOUND_INF && x_u[i] < BOUND_INF);
    let l_scale: Vec<Number> = (0..n)
        .map(|i| {
            let w = x_u[i] - x_l[i];
            if has_box && w > 0.0 { w } else { 1.0 }
        })
        .collect();

    let max_solves = cfg.max_solves_or_default();
    // Sample ceiling for solve-gated strategies (MLSL): one round of samples
    // per unit of solve budget. The patience-on-stall rule normally
    // terminates first; this guarantees `max_solves` bounds wall-clock even
    // when the clustering filter rejects everything (pounce#103).
    let batch = cfg.samples_per_round.unwrap_or(20).max(1);
    let max_samples = cfg
        .max_samples
        .unwrap_or_else(|| max_solves.saturating_mul(batch));

    let has_hessian = nnz_h > 0 && solver.has_exact_hessian();
    let mut driver = Driver {
        solver,
        hooks,
        trace: Vec::new(),
        base: Rc::clone(base),
        cfg,
        n,
        m,
        nnz_h,
        has_hessian,
        index_offset,
        x0,
        x_l: x_l.clone(),
        x_u: x_u.clone(),
        g_l,
        g_u,
        has_box,
        l_scale: l_scale.clone(),
        sampler: Sampler::new(cfg.seed, cfg.sobol),
        archive: Archive::new(cfg.dedup, l_scale.clone()),
        stagnant: 0,
        infeasible_landings: Vec::new(),
        n_solves: 0,
        max_solves,
        n_samples: 0,
        max_samples,
        psd_skipped_logged: false,
        warnings: Vec::new(),
        grad_ref,
        n_kkt_rejected: 0,
    };

    let stop = driver.run();

    // Rank the found minima by objective (best first).
    let minima: Vec<Minimum> = driver
        .archive
        .order_by_objective()
        .iter()
        .map(|&i| Minimum {
            x: driver.archive.xs[i].clone(),
            objective: driver.archive.fs[i],
            lambda: driver.archive.ls[i].clone(),
            z_l: driver.archive.zls[i].clone(),
            z_u: driver.archive.zus[i].clone(),
        })
        .collect();
    Ok(MinimaResult {
        minima,
        stop,
        n_solves: driver.n_solves,
        n_kkt_rejected: driver.n_kkt_rejected,
        scale: l_scale,
        warnings: driver.warnings,
        trace: driver.trace,
    })
}

/// Per-dimension width vector from a knob spec, mirroring
/// `_resolve_lengths`: a vector ⇒ used as given; a scalar ⇒ isotropic;
/// neither ("auto") ⇒ `frac · L` when a box is known, else `fallback`.
fn resolve_lengths(
    per_dim: Option<&Vec<f64>>,
    spec: Option<f64>,
    frac: f64,
    fallback: f64,
    l_scale: &[Number],
    has_box: bool,
) -> Vec<Number> {
    if let Some(v) = per_dim {
        return v.clone();
    }
    match spec {
        Some(s) => vec![s; l_scale.len()],
        None if has_box => l_scale.iter().map(|&l| frac * l).collect(),
        None => vec![fallback; l_scale.len()],
    }
}

/// Flooding's escape height for a bump of widths `sigma` on a minimum with
/// dense objective Hessian `h` (row-major `n×n`): `margin · μ_min` of
/// `S = diag(σ)·H·diag(σ)`, the smallest generalized eigenvalue of
/// `H v = μ·diag(1/σ²) v` — the height at which the bump turns the minimum
/// into a saddle — floored at `1e-12`. `None` if the eigensolve fails.
fn auto_amplitude(h: &[Number], sigma: &[Number], margin: f64) -> Option<f64> {
    let n = sigma.len();
    let mut s_mat = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            s_mat[i * n + j] = sigma[i] * sigma[j] * h[i * n + j];
        }
    }
    let mut w = vec![0.0; n];
    let mut v = vec![0.0; n * n];
    if !pounce_linalg::symmetric_eigen(&s_mat, n, &mut w, &mut v) {
        return None;
    }
    let mu_min = w.iter().cloned().fold(f64::INFINITY, f64::min);
    Some(margin * mu_min.max(1e-12))
}

/// The `kkt_tol` acceptance bound (gh#983 review): `kkt_tol · max(1, g0,
/// dual_scale)`, with `g0 = max |∇f(x0)|` and `dual_scale` the magnitude of the
/// terms the unscaled residual is assembled from at the returned point
/// (`final_unscaled_dual_scale`; `1` when not computed). Both are needed: at an
/// interior minimum those terms *are* `∇f`, which is ~0 there, so the second
/// alone reads the residual against itself. An absolute `1e-4` rejected every
/// candidate of an objective whose gradients live at `1e12`. On an O(1) model
/// the bound is the absolute one.
fn kkt_bound(kkt_tol: Number, dual_scale: Number, grad_ref: Number) -> Number {
    let scale = if dual_scale.is_finite() {
        dual_scale
    } else {
        1.0
    };
    kkt_tol * 1.0_f64.max(scale).max(grad_ref)
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
    use super::{
        auto_amplitude, better_sample_within, coord_in_bounds, kkt_bound, resolve_lengths,
        scaled_distance,
    };

    /// `_resolve_lengths`' cases (ported from `test_minima_scaling.py`).
    #[test]
    fn resolve_lengths_auto_scalar_vector() {
        let l = [10.0, 0.1];
        let close = |a: Vec<f64>, b: [f64; 2]| {
            assert!(
                a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12),
                "{a:?} vs {b:?}"
            )
        };
        // auto -> frac * range, per dimension
        close(resolve_lengths(None, None, 0.1, 0.5, &l, true), [1.0, 0.01]);
        // no box -> fallback
        close(
            resolve_lengths(None, None, 0.1, 0.5, &[1.0, 1.0], false),
            [0.5, 0.5],
        );
        // scalar -> isotropic
        close(
            resolve_lengths(None, Some(0.3), 0.1, 0.5, &l, true),
            [0.3, 0.3],
        );
        // vector -> as given, over the scalar
        let v = vec![2.0, 4.0];
        close(
            resolve_lengths(Some(&v), Some(0.3), 0.1, 0.5, &l, true),
            [2.0, 4.0],
        );
    }

    /// `_auto_amplitude`'s cases (ported from `test_minima_scaling.py`).
    #[test]
    fn auto_amplitude_from_curvature() {
        let h = [1.0, 0.0, 0.0, 100.0];
        // Isotropic σ: the smallest eigenvalue is 1, so amp = margin·1.
        let a = auto_amplitude(&h, &[1.0, 1.0], 2.0).unwrap();
        assert!((a - 2.0).abs() < 1e-12, "{a}");
        // A narrow σ in the stiff direction rebalances the metric (S = I).
        let a = auto_amplitude(&h, &[1.0, 0.1], 2.0).unwrap();
        assert!((a - 2.0).abs() < 1e-10, "{a}");
        // A non-positive curvature floors at 1e-12, not a negative height.
        let a = auto_amplitude(&[-1.0, 0.0, 0.0, 1.0], &[1.0, 1.0], 2.0).unwrap();
        assert_eq!(a, 2e-12);
    }

    /// The bound is relative to a *gradient* scale, never to the residual: a
    /// residual of 0.5 against scale 1 is rejected, against 1e6 it is not
    /// (the cases of `test_find_minima_relative_bound_still_rejects_a_non_
    /// stationary_point` in the Python suite).
    #[test]
    fn kkt_bound_is_relative_to_the_gradient_scale() {
        assert!(0.5 > kkt_bound(1e-4, 1.0, 0.0));
        assert!(0.5 <= kkt_bound(1e-4, 1e6, 0.0));
        assert!(0.5 <= kkt_bound(1e-4, 1.0, 1e6));
        // Floored at 1, and a NaN scale (path that does not compute it) is 1.
        assert_eq!(kkt_bound(1e-4, 1e-9, 0.0), 1e-4);
        assert_eq!(kkt_bound(1e-4, f64::NAN, 0.0), 1e-4);
    }

    /// The pruned sweep must give exactly the naive single-linkage answer.
    #[test]
    fn pruned_nearest_better_matches_the_naive_scan() {
        let mut state = 12345u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let l = [6.0, 4.0, 1.0];
        let xs: Vec<Vec<f64>> = (0..400)
            .map(|_| vec![6.0 * rnd() - 3.0, 4.0 * rnd() - 2.0, rnd()])
            .collect();
        let fs: Vec<f64> = (0..400).map(|_| rnd()).collect();
        for radius in [0.01, 0.1, 0.3, 2.0] {
            let fast = better_sample_within(&xs, &fs, &l, radius);
            for i in 0..xs.len() {
                let naive = (0..xs.len()).any(|j| {
                    j != i && fs[j] < fs[i] && scaled_distance(&xs[i], &xs[j], &l) < radius
                });
                assert_eq!(fast[i], naive, "radius {radius}, point {i}");
            }
        }
    }

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
