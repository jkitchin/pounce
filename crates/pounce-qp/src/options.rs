//! Per-solve options. The defaults mirror the §7.1 option-registry
//! values from the design note so the SQP-side wiring can forward
//! `OptionsList` entries straight through without translation.

use pounce_common::{Number, OptionsList, SolverException, option_invalid};
use std::time::Duration;

/// Active-set QP algorithm variant. Phase 5a ships only the sparse
/// parametric active-set method; other entries are placeholders to
/// keep the option name `sqp_qp_solver` stable as future variants
/// (e.g., a dense Goldfarb-Idnani for tiny dense QPs) appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QpAlgorithm {
    /// Sparse Schur-complement parametric active-set (§4.2,
    /// Kirches 2011 / Janka 2017). Default and only option in
    /// Phase 5a.
    #[default]
    ParametricActiveSet,
}

/// Anti-cycling rule. `Expand` is the SOTA default (§4.4,
/// Gill-Murray-Saunders-Wright 1989); `Bland` is a slower
/// guaranteed-finite fallback used in unit tests; `None` disables
/// anti-cycling and is for benchmarking only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AntiCyclingChoice {
    #[default]
    Expand,
    Bland,
    None,
}

#[derive(Debug, Clone)]
pub struct QpOptions {
    /// Solve-wide wall-clock budget shared by homotopy, elastic phase-1,
    /// feasibility/recovery solves, and seeded retries.
    pub time_limit: Option<Duration>,
    pub algorithm: QpAlgorithm,
    pub max_iter: u32,
    pub feas_tol: Number,
    pub opt_tol: Number,
    /// Maximum number of Schur-complement rank-1 updates before a
    /// fresh base-KKT refactorization. Default 50 per the design
    /// note §4.2 / §7.1; bound on the worst-case dense-Schur cost.
    pub max_schur_updates_before_refactor: u32,
    pub anti_cycling: AntiCyclingChoice,
    /// Elastic-mode penalty γ (§4.3). Default 1e6; large enough that
    /// the elastic slacks vanish at the solution of any feasible QP
    /// the SQP outer loop is likely to generate, small enough not to
    /// dominate the Hessian conditioning.
    pub elastic_gamma: Number,
    /// 0 = silent, 1 = per-solve summary, 2 = per-iteration trace,
    /// 3+ = per-pivot detail. Matches pounce's existing
    /// `print_level` convention.
    pub print_level: u8,

    /// §4.5 inertia control: when an LDLᵀ factor of the active-set
    /// KKT reports the wrong inertia or near-singularity, retry
    /// with `H ← H + δ·I` (only on the original-variable block).
    /// `inertia_shift_initial` is the first δ tried; subsequent
    /// retries multiply δ by `inertia_shift_factor`; the loop
    /// gives up after `inertia_max_shifts` attempts.
    ///
    /// Default values match the IPOPT-style perturbation handler
    /// in `pounce-algorithm/src/kkt/perturbation_handler.rs`.
    pub inertia_shift_initial: Number,
    pub inertia_shift_factor: Number,
    pub inertia_max_shifts: u32,

    /// Opt in to the §4.2 sparse Schur-complement update path in
    /// `solve_general`. When `true`, the inner loop uses a cached
    /// factor of the fixed-dim K_max matrix and absorbs working-
    /// set changes as Sherman-Morrison-Woodbury rank-2 updates,
    /// refactoring only when the Schur block reaches
    /// `max_schur_updates_before_refactor`. When `false`
    /// (default), each iteration assembles a fresh active-set
    /// KKT and factors from scratch — algorithmically correct,
    /// noticeably slower on large warm-started workloads.
    pub use_schur_updates: bool,

    /// Trace the §4.2 parametric homotopy on a **cold** solve instead of
    /// starting the conventional phase-1/phase-2 scheme.
    ///
    /// Off in this crate's defaults, and **on** in the convex QP driver
    /// ([`pounce_convex::active_set`]) which is where it was measured. The
    /// distinction is deliberate: this flag is read by every consumer of
    /// `pounce-qp`, including the SQP outer loop's inner subproblem solves, and
    /// that path has *not* been benchmarked with the homotopy. Enabling it
    /// crate-wide changes those solves too and does break
    /// `sqp::tests::classify_working_set_reproduces_sqp_solver_output_on_convex_eq`.
    ///
    /// The homotopy is the algorithm
    /// this crate is named for and the one the design note assumes (§4.3 has
    /// phase-1 driving its elastic slacks to zero *as the homotopy proceeds*);
    /// the conventional phase-1/phase-2 scheme is the substitute that existed
    /// because the homotopy did not.
    ///
    /// Full Maros-Mészáros, 138 problems, 120 s cap, same binary:
    ///
    /// | | correct | solved-but-wrong | timeouts |
    /// |---|---|---|---|
    /// | conventional | 58/138 | none | 54 |
    /// | homotopy | **71/138** | none | 49 |
    ///
    /// The trade is not uniform: 20 problems are gained, 7 lost. Six of the
    /// losses are *large* instances that previously solved and now hit the time
    /// cap (`AUG2D`, `AUG2DC`, `CONT-050`, `CONT-100`, `DTOC3`, `STADAT3`) —
    /// the homotopy is slower there, not wrong. The seventh, `QSHARE2B`,
    /// completes its path but its corrector then exhausts its iterations.
    /// Set this to `false` to get the old behaviour on such a workload.
    ///
    /// Most of that cost turned out to be a *defect* rather than the method:
    /// the ratio test stepped over crossings it had found, and a stepped-over
    /// row can never be recovered, so the path wedged. gh #434 fixed it (see
    /// [`crate::homotopy::RatioTest`]) and re-measured — `AUG2D`, `AUG2DC` and
    /// `QSHARE2B` recover outright, cold paths reaching `t = 1` go 92 → 98 of
    /// 138, and the median completed path halves. `DTOC3` and the other long
    /// paths remain, and #434 records why no runtime guard is shipped for them:
    /// no threshold on (path steps, `t`) separates them from the gains.
    /// Full measurement: `dev-notes/issue-434-homotopy-cost.md`.
    ///
    /// See [`crate::homotopy`].
    pub use_homotopy: bool,

    /// §4.4 full EXPAND anti-cycling primal perturbation. Active
    /// only when `anti_cycling = Expand`. The Harris two-pass
    /// (c14) prevents cycling at non-degenerate vertices; these
    /// parameters add protection at truly degenerate (α = 0)
    /// vertices via a monotonically growing tolerance:
    ///
    /// - `expand_tol_initial` — starting τ at each reset.
    /// - `expand_tol_growth`  — per-iteration increment of τ.
    /// - `expand_tol_max`     — τ ceiling; on hitting it, snap
    ///   all active-bound primals exactly to their bounds and
    ///   reset τ to `expand_tol_initial`.
    ///
    /// Defaults are conservative — they ensure cycling protection
    /// kicks in only on pathological degeneracy. References:
    /// Gill-Murray-Saunders-Wright 1989 §4 (the EXPAND name and
    /// the τ-growth schedule); SNOPT defaults.
    pub expand_tol_initial: Number,
    pub expand_tol_growth: Number,
    pub expand_tol_max: Number,

    /// May this solve conclude that the QP is unbounded below?
    ///
    /// `true` (default) is the F2/N1 behaviour described on
    /// [`crate::solver`]: a feasible descent direction that nothing
    /// blocks and that the model falls forever along is returned as
    /// `QpStatus::Unbounded` with the recession ray attached.
    ///
    /// `false` asks for a **point instead of a verdict**. The
    /// certification is skipped and the unblocked direction takes the
    /// δ-shifted proximal step (`α = 1`, the minimizer of
    /// `q(y) + ½δ‖y − x‖²`) so the inner loop keeps going. The solve
    /// then exits `Optimal` — at the proximal fixed point — or
    /// `MaxIter`, but never `Unbounded`.
    ///
    /// This exists for the SQP outer loop (gh #423). The *step* QP of a
    /// nonconvex NLP is unbounded below at every indefinite iterate that
    /// has nothing to block a negative-curvature direction — which, with
    /// `m = 0` and no finite bounds, is every indefinite iterate there
    /// is. That is a statement about the linearization, not about the
    /// NLP: the driver re-tests the ray against the true problem
    /// (`ray_certifies_unbounded`), and when the NLP turns out to be
    /// bounded it needs a usable step out of the same subproblem rather
    /// than a second unboundedness claim. Regularizing the model is the
    /// textbook answer there (Nocedal-Wright §18.4), and δ from §4.5
    /// inertia control is already exactly that regularization — so the
    /// re-solve just declines the certificate and keeps the shift.
    ///
    /// Leave this `true` for a standalone `solve_qp`: the *whole point*
    /// of solving a QP is to learn whether it has a minimizer.
    pub certify_recession_ray: bool,

    /// Check the **second-order** conditions before returning
    /// [`QpStatus::Optimal`](crate::QpStatus::Optimal) on a problem whose
    /// `H` is not claimed PSD, and escape along negative curvature when the
    /// check produces a witness (gh #848).
    ///
    /// Every `Optimal` exit in the engine is first-order: the projected
    /// gradient vanishes and the working set's multipliers carry admissible
    /// signs. On an indefinite `H` a *saddle point* satisfies exactly that,
    /// and the box-constrained path starts from the projection of the origin
    /// — which on `½xᵀ[[1,5];[5,1]]x` over `[−1,1]²` *is* the saddle, so the
    /// solve certified `obj = 0` against a true minimum of `−4`. See
    /// [`crate::negcurv`] for the test and its deliberate asymmetry: a
    /// rejection always carries a direction, an inconclusive probe always
    /// leaves the first-order verdict standing.
    ///
    /// `false` restores the pre-#848 behaviour exactly, including on the
    /// indefinite arm. It is not a performance knob: with
    /// `HessianInertia::Psd` the whole path is skipped anyway, so the convex
    /// arm pays nothing either way.
    ///
    /// `true` in [`QpOptions::default`], which is what every standalone entry
    /// point uses — `solver_selection=qp-active-set`,
    /// `pounce.qp.solve_qp(method="active-set")`, `ParametricActiveSetSolver`
    /// called directly. Those are the entry points gh #848 reports, and there
    /// the QP *is* the question being asked. `false` in
    /// [`QpOptions::sqp_subproblem`], which is a different question — see
    /// there.
    pub certify_second_order: bool,
    /// How many negative-curvature escapes one `solve` may take before it
    /// gives up and reports the budget rather than the point.
    ///
    /// Each escape adds a blocking row or bound to the working set, so a
    /// monotone run terminates in at most `n + m` of them; the cap is what
    /// bounds the non-monotone case, where the re-solve drops back out of the
    /// set it was just pushed into. Exhausting it downgrades the solve to
    /// `MaxIter` — never to `Optimal`, which is the defect this exists to
    /// prevent.
    pub neg_curv_max_escapes: u32,
    /// Inverse-iteration steps allowed per second-order probe.
    ///
    /// Each is one back-substitution against a factor that already exists
    /// plus one `H·d` product. The iteration converges at the ratio of the
    /// two smallest eigenvalues of `Zᵀ(H + δI)Z`, and `δ` comes off the §4.5
    /// ladder, so it overshoots `|λ_min|` by at most `inertia_shift_factor`;
    /// the worst-case ratio that leaves is about
    /// `(λ_max + 100|λ_min|)/(99|λ_min|)`. Termination is on the *sign* of
    /// `dᵀHd`, not on convergence, which is reached long before the
    /// eigenvector is.
    pub neg_curv_probe_iters: u32,
    /// Geometric bisections used to tighten the inertia shift before the
    /// probe iterates.
    ///
    /// The §4.5 ladder brackets `|λ_min|` between the rung that worked and
    /// the one before it, a factor of `inertia_shift_factor` apart. Each
    /// refinement halves that bracket on a log scale at the cost of one
    /// factorization, leaving an overshoot of `100^(2^−r)` — 1.8% at the
    /// default 8. The overshoot is what sets the inverse iteration's
    /// convergence rate, so paying here is what keeps `neg_curv_probe_iters`
    /// small: reusing the ladder's own shift instead leaves rates as bad as
    /// 1.02 per step on `H = diag(1, −1)`, where the ladder stops at `δ = 100`.
    pub neg_curv_shift_refinements: u32,
    /// Relative margin a curvature must clear to count as negative:
    /// `dᵀHd < −neg_curv_tol · ‖H‖∞` at `‖d‖₂ = 1`.
    ///
    /// Relative because the verdict must not depend on the units of the
    /// objective. The margin is a *rejection* threshold, so erring large is
    /// the safe direction: it declines to reject a curvature that could be
    /// rounding noise, and an unrejected point keeps the status the engine
    /// already assigned it.
    pub neg_curv_tol: Number,
}

impl Default for QpOptions {
    fn default() -> Self {
        Self {
            time_limit: None,
            algorithm: QpAlgorithm::default(),
            max_iter: 200,
            feas_tol: 1e-9,
            opt_tol: 1e-9,
            max_schur_updates_before_refactor: 50,
            anti_cycling: AntiCyclingChoice::default(),
            elastic_gamma: 1e6,
            print_level: 0,
            inertia_shift_initial: 1e-8,
            inertia_shift_factor: 100.0,
            inertia_max_shifts: 12,
            use_schur_updates: false,
            use_homotopy: false,
            expand_tol_initial: 1e-12,
            expand_tol_growth: 1e-11,
            expand_tol_max: 1e-7,
            certify_recession_ray: true,
            certify_second_order: true,
            neg_curv_max_escapes: 20,
            neg_curv_probe_iters: 20,
            neg_curv_shift_refinements: 8,
            neg_curv_tol: 1e-8,
        }
    }
}

impl QpOptions {
    /// The defaults for a QP solved as an SQP *step subproblem*, rather than
    /// as a question in its own right.
    ///
    /// Identical to [`QpOptions::default`] except that
    /// [`certify_second_order`](Self::certify_second_order) is off, and the
    /// reason is not caution — it is that the two callers are asking different
    /// questions of the same engine.
    ///
    /// A standalone `solve_qp` asks *where is this QP's minimum*, and a
    /// first-order point that is not one is a wrong answer (gh #848). An SQP
    /// step QP asks *give me a step*, about a local model built at the current
    /// iterate from the current multiplier estimates — and that model's
    /// second-order verdict is not the NLP's. HS071 is the counterexample and
    /// it is not exotic: at iteration 0 the multipliers are still zero, so
    /// `∇²L` is `∇²f`, whose reduced Hessian on the working set's null space
    /// is negative (`dᵀHd = -4.05e-2`) at a point that *is* a local minimum of
    /// the NLP. Started at `x*`, certification sends the engine chasing that
    /// curvature to a box bound and back — five outer iterations where one
    /// sufficed, and `QpIterationLimit` at iteration 0 from `x* + 1e-8·e₀`,
    /// which is every warm start (gh #484). Modifying the Hessian, not
    /// following the curvature, is the textbook answer for an SQP step
    /// (Nocedal-Wright §18.4), and §4.5's δ already is that modification.
    ///
    /// Turning it on for the SQP is a supported experiment —
    /// `sqp_qp_certify_second_order=yes` — and it does fix real wrong answers
    /// there: `algorithm=active-set-sqp` on `nonconvex_qp.nl` stops reporting
    /// the constrained *maximum* as `Solve_Succeeded`. Making it the default
    /// needs the Hessian-modification path first; that is gh #856, not this
    /// change.
    #[must_use]
    pub fn sqp_subproblem() -> Self {
        Self {
            certify_second_order: false,
            ..Self::default()
        }
    }
}

/// Explicitly-set `sqp_qp_*` overrides for a [`QpOptions`] base.
///
/// Every field is `None` unless the user set the corresponding option
/// *explicitly*, so a caller can tell "left at the default" from "asked for
/// the default value". Both consumers need that distinction, and for the same
/// reason: each starts from a base that is not [`QpOptions::default`] and an
/// explicit request has to win over it. `pounce-convex`'s direct active-set
/// driver picks a size-scaled `max_iter` and turns Schur updates on;
/// `pounce-algorithm`'s SQP path starts from the plain defaults but must not
/// overwrite them with zeros for options nobody set.
///
/// The type exists because these knobs became unreachable when
/// `solver_selection=qp-active-set` moved off the SQP outer loop: the
/// `sqp_qp_*` family was introduced for the QP subproblem *of that loop*, and
/// with no SQP in the picture every one of them silently became a no-op on the
/// direct convex route. The spelling is kept because it is the documented,
/// already-in-use name; only the delivery route changed.
///
/// This lives in `pounce-qp` because [`QpOptions`] does: it is the one crate
/// both readers already depend on (`pounce-algorithm` has no `pounce-convex`
/// dependency), so it is the only place a single copy can serve both. It was
/// two copies until then — `pounce_convex::ActiveSetOverrides` for the direct
/// driver and `pounce_algorithm::application::apply_qp_subproblem_options` for
/// the SQP subproblem — reading the same eight registered names into the same
/// struct, and they had already drifted apart on how they treat a value the
/// registry would have rejected.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ActiveSetOverrides {
    pub max_iter: Option<u32>,
    pub anti_cycling: Option<AntiCyclingChoice>,
    pub feas_tol: Option<Number>,
    pub opt_tol: Option<Number>,
    pub elastic_gamma: Option<Number>,
    pub use_schur_updates: Option<bool>,
    pub use_homotopy: Option<bool>,
    pub max_schur_updates_before_refactor: Option<u32>,
    pub certify_second_order: Option<bool>,
}

impl ActiveSetOverrides {
    /// True when the caller set nothing.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Overlay the explicitly-set values onto `o`, leaving the rest alone.
    pub fn apply(&self, o: &mut QpOptions) {
        if let Some(v) = self.max_iter {
            o.max_iter = v;
        }
        if let Some(v) = self.anti_cycling {
            o.anti_cycling = v;
        }
        if let Some(v) = self.feas_tol {
            o.feas_tol = v;
        }
        if let Some(v) = self.opt_tol {
            o.opt_tol = v;
        }
        if let Some(v) = self.elastic_gamma {
            o.elastic_gamma = v;
        }
        if let Some(v) = self.use_schur_updates {
            o.use_schur_updates = v;
        }
        if let Some(v) = self.use_homotopy {
            o.use_homotopy = v;
        }
        if let Some(v) = self.max_schur_updates_before_refactor {
            o.max_schur_updates_before_refactor = v;
        }
        if let Some(v) = self.certify_second_order {
            o.certify_second_order = v;
        }
    }

    /// Read the explicitly-set `sqp_qp_*` controls out of a registered
    /// [`OptionsList`].
    ///
    /// The bounds re-checked here duplicate the ones
    /// `pounce_algorithm::upstream_options` registers, so on any path that
    /// went through the registry they are unreachable — the registry rejects
    /// the value at *set* time. They exist for the caller who builds an
    /// `OptionsList` with no registry attached, where nothing else would.
    /// `pounce-cli`'s `convex_option_readers_match_the_registry` test pins the
    /// two sets of bounds together so the pair cannot drift.
    pub fn try_from_options_list(options: &OptionsList) -> Result<Self, SolverException> {
        let mut parsed = Self::default();

        let (max_iter, explicitly_set) = options.get_integer_value("sqp_qp_max_iter", "")?;
        if explicitly_set {
            let value = u32::try_from(max_iter).map_err(|_| {
                option_invalid("sqp_qp_max_iter", "must be positive and fit in u32")
            })?;
            if value == 0 {
                return Err(option_invalid("sqp_qp_max_iter", "must be positive"));
            }
            parsed.max_iter = Some(value);
        }

        let (anti_cycling, explicitly_set) = options.get_string_value("sqp_qp_anti_cycling", "")?;
        if explicitly_set {
            parsed.anti_cycling = Some(match anti_cycling.as_str() {
                "bland" => AntiCyclingChoice::Bland,
                "expand" => AntiCyclingChoice::Expand,
                "none" => AntiCyclingChoice::None,
                _ => {
                    return Err(option_invalid(
                        "sqp_qp_anti_cycling",
                        format_args!("unknown value \"{anti_cycling}\""),
                    ));
                }
            });
        }

        let (feas_tol, explicitly_set) = options.get_numeric_value("sqp_qp_feas_tol", "")?;
        if explicitly_set {
            if feas_tol <= 0.0 || feas_tol.is_nan() {
                return Err(option_invalid(
                    "sqp_qp_feas_tol",
                    "must be greater than zero",
                ));
            }
            parsed.feas_tol = Some(feas_tol);
        }

        let (opt_tol, explicitly_set) = options.get_numeric_value("sqp_qp_opt_tol", "")?;
        if explicitly_set {
            if opt_tol <= 0.0 || opt_tol.is_nan() {
                return Err(option_invalid(
                    "sqp_qp_opt_tol",
                    "must be greater than zero",
                ));
            }
            parsed.opt_tol = Some(opt_tol);
        }

        let (elastic_gamma, explicitly_set) =
            options.get_numeric_value("sqp_qp_elastic_gamma", "")?;
        if explicitly_set {
            if elastic_gamma <= 0.0 || elastic_gamma.is_nan() {
                return Err(option_invalid(
                    "sqp_qp_elastic_gamma",
                    "must be greater than zero",
                ));
            }
            parsed.elastic_gamma = Some(elastic_gamma);
        }

        // Two lookups rather than a bare `get_bool_value`, and deliberately so:
        // with no registry attached `get_string_value` answers "" / not-found
        // for an unset name, and `get_bool_value` would reject that "" as a
        // non-boolean instead of reporting it unset. Read the presence flag
        // first, and only decode a value that is actually there.
        let (_, explicitly_set) = options.get_string_value("sqp_qp_use_schur_updates", "")?;
        if explicitly_set {
            parsed.use_schur_updates =
                Some(options.get_bool_value("sqp_qp_use_schur_updates", "")?.0);
        }
        let (_, explicitly_set) = options.get_string_value("sqp_qp_use_homotopy", "")?;
        if explicitly_set {
            parsed.use_homotopy = Some(options.get_bool_value("sqp_qp_use_homotopy", "")?.0);
        }

        let (updates, explicitly_set) =
            options.get_integer_value("sqp_qp_max_schur_updates_before_refactor", "")?;
        if explicitly_set {
            let value = u32::try_from(updates).map_err(|_| {
                option_invalid(
                    "sqp_qp_max_schur_updates_before_refactor",
                    "must be positive and fit in u32",
                )
            })?;
            if value == 0 {
                return Err(option_invalid(
                    "sqp_qp_max_schur_updates_before_refactor",
                    "must be positive",
                ));
            }
            parsed.max_schur_updates_before_refactor = Some(value);
        }

        let (_, explicitly_set) = options.get_string_value("sqp_qp_certify_second_order", "")?;
        if explicitly_set {
            parsed.certify_second_order =
                Some(options.get_bool_value("sqp_qp_certify_second_order", "")?.0);
        }

        Ok(parsed)
    }
}

#[cfg(test)]
mod option_reader_tests {
    use super::*;
    use pounce_common::Index;

    fn set_num(options: &mut OptionsList, name: &str, value: Number) {
        options.set_numeric_value(name, value, true, true).unwrap();
    }

    fn set_int(options: &mut OptionsList, name: &str, value: Index) {
        options.set_integer_value(name, value, true, true).unwrap();
    }

    fn set_str(options: &mut OptionsList, name: &str, value: &str) {
        options.set_string_value(name, value, true, true).unwrap();
    }

    #[test]
    fn active_set_controls_materialize_only_explicit_overrides() {
        let defaults = ActiveSetOverrides::try_from_options_list(&OptionsList::new()).unwrap();
        assert!(defaults.is_empty());

        let mut options = OptionsList::new();
        set_int(&mut options, "sqp_qp_max_iter", 37);
        set_str(&mut options, "sqp_qp_anti_cycling", "bland");
        set_num(&mut options, "sqp_qp_feas_tol", 1e-7);
        set_num(&mut options, "sqp_qp_opt_tol", 2e-7);
        set_num(&mut options, "sqp_qp_elastic_gamma", 1e4);
        set_str(&mut options, "sqp_qp_use_schur_updates", "yes");
        set_str(&mut options, "sqp_qp_use_homotopy", "no");
        set_int(&mut options, "sqp_qp_max_schur_updates_before_refactor", 12);

        let parsed = ActiveSetOverrides::try_from_options_list(&options).unwrap();
        assert_eq!(parsed.max_iter, Some(37));
        assert_eq!(parsed.anti_cycling, Some(AntiCyclingChoice::Bland));
        assert_eq!(parsed.feas_tol, Some(1e-7));
        assert_eq!(parsed.opt_tol, Some(2e-7));
        assert_eq!(parsed.elastic_gamma, Some(1e4));
        assert_eq!(parsed.use_schur_updates, Some(true));
        assert_eq!(parsed.use_homotopy, Some(false));
        assert_eq!(parsed.max_schur_updates_before_refactor, Some(12));
    }

    /// `apply` is the half both consumers share: an empty set must leave a
    /// non-default base untouched, and an explicit value must beat it. That
    /// is the whole reason the type carries `Option`s — the direct convex
    /// driver seeds a size-scaled `max_iter` and turns Schur updates on, and
    /// a user who asked for neither must keep the seed.
    #[test]
    fn apply_overlays_only_what_was_set() {
        let seeded = QpOptions {
            max_iter: 4242,
            use_schur_updates: true,
            ..QpOptions::default()
        };

        let mut untouched = seeded.clone();
        ActiveSetOverrides::default().apply(&mut untouched);
        assert_eq!(untouched.max_iter, 4242);
        assert!(untouched.use_schur_updates);

        let mut overridden = seeded.clone();
        ActiveSetOverrides {
            max_iter: Some(9),
            ..ActiveSetOverrides::default()
        }
        .apply(&mut overridden);
        assert_eq!(overridden.max_iter, 9);
        // Untouched fields keep the seed, not `QpOptions::default()`.
        assert!(overridden.use_schur_updates);
    }

    /// Without a registry attached nothing validates at set time, so the
    /// reader is the only thing standing between a bad value and the solver.
    #[test]
    fn malformed_unregistered_values_are_rejected() {
        let mut options = OptionsList::new();
        set_int(&mut options, "sqp_qp_max_iter", 0);
        assert!(ActiveSetOverrides::try_from_options_list(&options).is_err());

        let mut options = OptionsList::new();
        set_str(&mut options, "sqp_qp_anti_cycling", "maybe");
        assert!(ActiveSetOverrides::try_from_options_list(&options).is_err());

        let mut options = OptionsList::new();
        set_num(&mut options, "sqp_qp_feas_tol", 0.0);
        assert!(ActiveSetOverrides::try_from_options_list(&options).is_err());

        let mut options = OptionsList::new();
        set_int(&mut options, "sqp_qp_max_schur_updates_before_refactor", 0);
        assert!(ActiveSetOverrides::try_from_options_list(&options).is_err());
    }

    /// The shared constructor is `#[track_caller]`, so a rejection points at
    /// the read site rather than at one line inside `pounce-common`.
    #[test]
    fn a_rejection_is_attributed_to_this_reader() {
        let mut options = OptionsList::new();
        set_int(&mut options, "sqp_qp_max_iter", 0);
        let err = ActiveSetOverrides::try_from_options_list(&options).unwrap_err();
        assert!(
            err.to_string().contains("options.rs"),
            "expected the reader's own file, got: {err}"
        );
    }
}
