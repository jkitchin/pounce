//! Two-dimensional filter — port of `Algorithm/IpFilter.{hpp,cpp}`.
//!
//! Stores `(theta, phi)` pairs and answers "is this point dominated
//! by any pair already in the filter?". Upstream's dominance rule
//! (`<=` not `<`), with a round-off floor on the `theta` axis that is zero
//! unless the driver's gh#945 retry pass sets it — see
//! [`Filter::dominated_by_any`].

use pounce_common::types::Number;

/// One filter entry — port of `Ipopt::FilterEntry`.
#[derive(Debug, Clone, Copy)]
pub struct FilterEntry {
    pub theta: Number,
    pub phi: Number,
    /// Iteration index this entry was added at — used for clearing
    /// on restoration return; mirrors upstream's `iter_` field.
    pub iter: i32,
}

impl FilterEntry {
    pub fn new(theta: Number, phi: Number, iter: i32) -> Self {
        Self { theta, phi, iter }
    }
}

/// `IpFilter.cpp:FilterEntry::Acceptable` — a point clears an entry when
/// it is no worse in *either* coordinate.
///
/// `theta_floor` is the round-off floor of evaluating the constraints at the
/// current iterate, or zero for upstream's bare `<=`; the `phi` arm is
/// always upstream's. Both the asymmetry and the fact that the floor is live
/// only during the gh#945 retry pass are measured, not aesthetic — see
/// [`Filter::dominated_by_any`].
#[inline]
fn entry_accepts(e: &FilterEntry, theta: Number, phi: Number, theta_floor: Number) -> bool {
    // The allowance is the round-off floor the driver measured at the
    // iterate, capped by the one every *other* acceptance test in this line
    // search already makes — `compare_le`'s `10·eps·max(1, |BasVal|)`. The
    // cap is what keeps a badly scaled model from having its `theta` axis
    // switched off wholesale: `theta`'s round-off bound is a product of
    // ∞-norms, so on a model whose largest Jacobian entry and largest
    // variable live in *different* rows it runs orders above the noise those
    // rows actually carry. Measured on `square_flowsheet_resto`, whose bound
    // reads 2.0e-7 against a `theta` pinned at 1.04e-9 jittering by 2e-14.
    let allowance = theta_floor.min(10.0 * Number::EPSILON * e.theta.abs().max(1.0));
    theta <= e.theta + allowance || phi <= e.phi
}

/// Two-dimensional filter. Backed by a `Vec` for predictable
/// iteration order — matches upstream `std::list<FilterEntry>`
/// insertion behavior.
#[derive(Debug, Default, Clone)]
pub struct Filter {
    entries: Vec<FilterEntry>,
    /// Round-off floor for the `theta` comparison (gh#945), in `theta`'s own
    /// units. **Zero by default**, which is upstream's rule exactly; the
    /// filter line search sets it per iteration from
    /// `IpoptCq::theta_evaluation_noise_floor`. See
    /// [`Filter::dominated_by_any`].
    theta_floor: Number,
}

impl Filter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> &[FilterEntry] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Set the gh#945 round-off floor for the `theta` axis, in `theta`'s own
    /// units. Zero is the default and is upstream's rule; the driver sets a
    /// nonzero value for the duration of one retry pass and clears it again.
    /// See [`Filter::dominated_by_any`].
    pub fn set_theta_roundoff_floor(&mut self, floor: Number) {
        self.theta_floor = if floor.is_finite() {
            floor.max(0.0)
        } else {
            0.0
        };
    }

    /// True if `(theta, phi)` is dominated by *any* existing entry.
    /// A point is **not acceptable** iff some entry beats it in BOTH
    /// coordinates:
    ///
    /// ```text
    ///   theta_new > e.theta   AND   phi_new > e.phi                 (default)
    ///   theta_new > e.theta + floor  AND  phi_new > e.phi
    ///                                 (during the gh#945 retry pass)
    /// ```
    ///
    /// This is the negation of `IpFilter.cpp:FilterEntry::Acceptable`,
    /// which returns true when *any* coord satisfies `vals[i] <= vals_[i]`.
    /// Exact ties stay acceptable, as upstream (`1.0 <= 1.0` → true at the
    /// first coord); a bare `>=` would over-reject and trigger spurious
    /// adaptive-μ free→fixed transitions when the iterate is approximately
    /// constant.
    ///
    /// **The `theta` floor is a pounce deviation from upstream (gh#945).**
    /// Upstream compares both coordinates raw. Every *other* acceptance test
    /// in this line search
    /// ([`super::filter_acceptor::FilterLsAcceptor::armijo_holds`],
    /// `is_sufficient_progress`, both axes) allows for round-off instead,
    /// for the reason recorded on `compare_le`: near a solution the
    /// quantities being compared are floating-point summation noise, and a
    /// bare comparison turns that noise into a verdict. The filter is the
    /// one test still reading raw values, and the `theta` axis is where that
    /// bites, because `theta` has a hard floor the barrier objective does
    /// not — on a model whose constraints are **linear**, or any model once
    /// it is feasible to round-off, every entry records a `theta` of a few
    /// `ulp` and later trials are ranked on which way the last sum rounded.
    ///
    /// Measured on gh#945's equality-constrained convex QP under
    /// `limited-memory`, at iteration 94, trial 3, `alpha = 0.125`:
    ///
    /// ```text
    ///   iterate  theta = 0.0                 phi = 9.35484845847691648e-1
    ///   trial    theta = 5.551115123125783e-16
    ///            phi   = 9.35484840739183920e-1   (a 5.1e-9 DECREASE)
    ///   entry    theta = 1.110211922394910e-16
    ///            phi   = 9.35484820369971715e-1
    /// ```
    ///
    /// The trial passes Armijo on that decrease and the filter rejects it
    /// anyway. Note which arm decides. The `phi` arm is doing real work —
    /// the trial is 2.0e-8 *above* the best barrier value the filter holds,
    /// a number rather than noise. The `theta` arm decides it alone, on
    /// `5.55e-16 > 1.11e-16`: a gap of 4.4e-16 between two roundings of the
    /// same identically-zero quantity. The α-loop then backtracks to
    /// `alpha_min`, hands a point *feasible to 0.0* to restoration — which
    /// has no violation to minimize — and the solve exits
    /// `Error_In_Step_Computation` from the optimum.
    ///
    /// Forgiving that is not a weakening of the rule: the two `theta`s are
    /// *tied*, both being zero, so the filter has no feasibility ground to
    /// stand on, and upstream's own `theta <= e.theta` would say so if the
    /// values it compared were not noise-perturbed.
    ///
    /// **The floor is not on for every filter decision, and that is the
    /// whole design.** Nothing here turns it on: it is zero unless
    /// [`super::backtracking::BacktrackingLineSearch::run_filter_line_search`]
    /// sets it, which it does on exactly one retry pass, after the α-loop
    /// has already run out and only when the iterate it ran out at is
    /// feasible to its own round-off — i.e. when the restoration phase the
    /// driver is about to call has nothing to minimize. Applying the floor
    /// to *every* decision instead was implemented and measured, and it is
    /// wrong: on MacMPEC's `qpec_small` it flips three decisions inside the
    /// same round-off band and costs the model its answer at a floor of
    /// `4.44e-16` — precisely what the gh#945 trial above requires — with no
    /// constant and no scale-aware formula separating the two, since both
    /// models sit at `‖x‖∞ ≈ 1` with `theta` at round-off. What *does*
    /// separate them is the question the retry gate asks, because it is a
    /// question about the iterate rather than about a pair of entries:
    /// `qpec_small` under the `prod_eq` lowering fails its line search at
    /// `theta = 1.746e-15` against its own floor of `6.66e-16`, 2.6× above
    /// it, so restoration there has real violation to work on, the gate
    /// declines, and that model's trajectory is byte-identical. See
    /// `pounce-algorithm/tests/issue_884_biactive_dual_divergence.rs`,
    /// `the_gh945_retry_gate_is_what_this_fixture_relies_on`.
    ///
    /// The gate is not a promise to leave every MPCC alone, and where it
    /// does fire on one the result is the opposite of the always-on floor's.
    /// The CLI reproducer `mpcc_qpec_small_biactive` under
    /// `bound_relax_factor=0 mu_strategy_fallback=no` reaches it, and its
    /// base solve stops needing gh#884's `perturb_always_cd` retry: it
    /// converges on its own at `Optimal`/100, KKT error `5.65e-10` (unscaled
    /// dual infeasibility `6.26e-10`), against the
    /// promoted `9.96e-8` it used to have to buy
    /// (`pounce-cli/tests/issue884_promotion_gate_reads_the_answer.rs`,
    /// `the_reproducer_no_longer_needs_the_retry`).
    ///
    /// **The allowance is capped**, and [`entry_accepts`] carries why: a
    /// round-off bound built from ∞-norms runs orders above the noise the
    /// near-zero rows actually carry whenever the largest Jacobian entry and
    /// the largest variable live in different rows. `square_flowsheet_resto`
    /// reads a bound of 2.0e-7 against a `theta` pinned at 1.04e-9 — five
    /// orders above round-off — and uncapped it loses its `Solve_Succeeded`
    /// on the *exact* leg. The driver's gate reads the **same** capped
    /// expression, evaluated at the iterate rather than at the entry, so that
    /// model's retry never opens at all and its trajectory is byte-identical
    /// (gh#946 review; capping only the allowance and not the gate left it
    /// as the second of two moved sweep legs).
    ///
    /// **The `phi` arm never gets a floor**, on its own measurement:
    /// `10·eps·max(1, |phi|)` is an *absolute* number once `|phi|` is large
    /// — on `mu_fallback_point_floor` (`phi ~ 3.8e7`) it lets the filter
    /// accept a barrier increase of 8.5e-8, a real weakening of the
    /// globalization — and it costs `autocorr_bern55-06` its status
    /// (`SolveSucceeded`/58 to `Solved_To_Acceptable_Level`/1316).
    pub fn dominated_by_any(&self, theta: Number, phi: Number) -> bool {
        self.entries
            .iter()
            .any(|e| !entry_accepts(e, theta, phi, self.theta_floor))
    }

    /// Add the entry and prune any existing entries that the new one
    /// strictly dominates. Mirrors `IpFilter::AddEntry`.
    pub fn add(&mut self, theta: Number, phi: Number, iter: i32) {
        self.entries.retain(|e| !(theta <= e.theta && phi <= e.phi));
        self.entries.push(FilterEntry::new(theta, phi, iter));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_dominates_nothing() {
        let f = Filter::new();
        assert!(!f.dominated_by_any(1.0, 1.0));
    }

    #[test]
    fn add_entry_then_dominated_check() {
        let mut f = Filter::new();
        f.add(1.0, 5.0, 0);
        // Strictly worse in both → dominated.
        assert!(f.dominated_by_any(2.0, 6.0));
        // Exact tie → acceptable per upstream (`vals[i] <= vals_[i]`
        // satisfied at i=0 → entry returns Acceptable=true).
        assert!(!f.dominated_by_any(1.0, 5.0));
        // Smaller in either dimension → not dominated.
        assert!(!f.dominated_by_any(0.5, 5.0));
        assert!(!f.dominated_by_any(1.0, 4.9));
    }

    #[test]
    fn add_entry_prunes_dominated_existing_entries() {
        let mut f = Filter::new();
        f.add(2.0, 6.0, 0);
        f.add(1.0, 5.0, 1);
        // The first entry was dominated by the second and should
        // have been pruned.
        assert_eq!(f.entries().len(), 1);
        assert_eq!(f.entries()[0].iter, 1);
    }

    /// gh#945, the exact numbers from the failing trajectory. A filter
    /// entry recorded at a `theta` of one `ulp` must not rank a trial whose
    /// `theta` is two `ulp` as worse — that difference is which way the
    /// constraint sum rounded, not a feasibility verdict. Reverting
    /// `entry_accepts`'s `theta` arm to a bare `<=` makes this dominated,
    /// which is the whole defect.
    #[test]
    fn theta_at_roundoff_scale_does_not_dominate() {
        let mut f = Filter::new();
        f.set_theta_roundoff_floor(2.220446049250313e-15);
        // The filter entry that did the rejecting, read out of the failing
        // solve: theta at half an ulp of 1.0, phi the scaled barrier
        // objective of an earlier and better iterate.
        f.add(1.110211922394910e-16, 9.35484820369971715e-1, 0);
        // The trial at alpha = 0.125, iteration 94. It decreases phi by
        // 5.1e-9 from the current iterate and passes Armijo, but sits 2.0e-8
        // above this entry's phi — so the phi arm cannot clear it and the
        // whole decision is the theta arm's. And that arm is comparing
        // 5.55e-16 against 1.11e-16: two roundings of the same zero.
        assert!(!f.dominated_by_any(5.551115123125783e-16, 9.35484840739183920e-1));
    }

    /// The floor is a round-off allowance, not a tolerance: a `theta` that
    /// is genuinely worse still dominates. Without this the fix would be
    /// switching the filter's `theta` axis off rather than denoising it.
    #[test]
    fn theta_above_the_roundoff_band_still_dominates() {
        let mut f = Filter::new();
        f.set_theta_roundoff_floor(2.220446049250313e-15);
        f.add(1e-10, 1.0, 0);
        // 1e-9 is a hundred times the entry and far above 10·eps.
        assert!(f.dominated_by_any(1e-9, 2.0));
        // Still inside the band relative to `max(1, |e.theta|)`.
        assert!(!f.dominated_by_any(1e-10 + 1e-15, 2.0));
    }

    /// The `phi` axis keeps upstream's bare comparison. Pinned because
    /// giving it a floor too was measured and rejected (see
    /// `Filter::dominated_by_any`), and a later "make it symmetric" edit
    /// would silently re-take that decision.
    #[test]
    fn phi_axis_carries_no_slack() {
        let mut f = Filter::new();
        f.set_theta_roundoff_floor(2.220446049250313e-15);
        f.add(0.0, 1.0, 0);
        // theta strictly above the band and phi above the entry by one
        // ulp of 1.0 → dominated, exactly as upstream.
        assert!(f.dominated_by_any(1.0, 1.0 + f64::EPSILON));
        // Equal phi is still acceptable (upstream's `<=`).
        assert!(!f.dominated_by_any(1.0, 1.0));
    }

    /// The floor is **off unless the driver sets it**. A default `Filter`
    /// must reproduce upstream's rule exactly: the retry pass is the only
    /// place a nonzero floor is ever live, and it clears it again on the way
    /// out. Pinned because a floor left switched on for every decision costs
    /// `qpec_small` its certificate — see `Filter::dominated_by_any`.
    #[test]
    fn the_floor_is_off_by_default() {
        let mut f = Filter::new();
        f.add(1.110211922394910e-16, 9.35484820369971715e-1, 0);
        let (theta, phi) = (5.551115123125783e-16, 9.35484840739183920e-1);
        assert!(
            f.dominated_by_any(theta, phi),
            "a default Filter applied the gh#945 floor; it must be upstream's \
             bare comparison until the retry pass sets one"
        );
        f.set_theta_roundoff_floor(2.220446049250313e-15);
        assert!(!f.dominated_by_any(theta, phi));
        f.set_theta_roundoff_floor(0.0);
        assert!(
            f.dominated_by_any(theta, phi),
            "clearing the floor did not restore upstream's comparison; the \
             driver clears it after the retry pass and the next pass must \
             not inherit it"
        );
    }

    /// The allowance is **capped** at `compare_le`'s own round-off
    /// allowance, `10·eps·max(1, |e.theta|)`, however large a floor the
    /// driver hands over.
    ///
    /// `theta`'s round-off bound is a product of ∞-norms, so on a model
    /// whose largest Jacobian entry and largest variable sit in different
    /// rows it runs orders above the noise the near-zero rows carry. Left
    /// uncapped it switches the `theta` axis off wholesale there: measured
    /// on `square_flowsheet_resto`, whose bound reads 2.0e-7 against a
    /// `theta` pinned at 1.04e-9 jittering by 2e-14, and which loses its
    /// `Solve_Succeeded` on the **exact** leg of
    /// `scripts/sweep-fixtures.sh` without this line.
    #[test]
    fn the_allowance_is_capped_at_the_compare_le_band() {
        let mut f = Filter::new();
        f.set_theta_roundoff_floor(2.0e-7);
        f.add(1.041403e-9, 1.0, 0);
        // Inside `compare_le`'s band: still forgiven.
        assert!(!f.dominated_by_any(1.041403e-9 + 1e-15, 2.0));
        // Two orders below the floor the driver handed over, and still far
        // above 10·eps: the cap is what makes this dominated.
        assert!(f.dominated_by_any(1.041403e-9 + 1e-9, 2.0));
    }

    #[test]
    fn pareto_set_is_preserved() {
        let mut f = Filter::new();
        f.add(1.0, 5.0, 0);
        f.add(2.0, 3.0, 1); // smaller phi, larger theta → not dominated
        f.add(3.0, 1.0, 2);
        assert_eq!(f.entries().len(), 3);
    }
}
