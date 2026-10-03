//! Primal-dual perturbation handler — port of
//! `Algorithm/IpPDPerturbationHandler.{hpp,cpp}`.
//!
//! Lives in `pounce-common` so both KKT consumers can use it: the NLP
//! filter-IPM (`pounce-algorithm`) and the active-set QP (`pounce-qp`). The
//! dependency runs `pounce-algorithm -> pounce-qp`, so a shared home is the
//! only way for the QP side to reach it without duplication.
//!
//! Owns the four perturbations `(δ_x, δ_s, δ_c, δ_d)` that the
//! `PDFullSpaceSolver` adds to the augmented system to recover correct
//! inertia / non-singularity. Implements upstream's full state
//! machine:
//!
//! * [`Self::consider_new_system`] — first call per new aug-system.
//!   Finalizes the previous trial's degeneracy probe, decides whether
//!   to start a new degeneracy test, and seeds `δ_c` / `δ_d` if the
//!   Jacobian is already known to be degenerate (or `perturb_always_cd`
//!   is on).
//! * [`Self::perturb_for_singular`] — escalation step taken when MA57
//!   reports `Singular`.
//! * [`Self::perturb_for_wrong_inertia`] — escalation step taken when
//!   the factor's negative-eigenvalue count disagrees with what the
//!   KKT structure requires.
//! * [`Self::current_perturbation`] — read the most recently committed
//!   `(δ_x, δ_s, δ_c, δ_d)`.
//!
//! Returns `false` when no further escalation is possible (caller must
//! enter the restoration phase). The `info_string`-mutation calls in
//! upstream are emitted via the `IpoptData` handle the caller passes
//! in; if `None` is passed, the strings are simply dropped.

use crate::types::{Index, Number};

/// Sink for the two diagnostic writes the handler performs.
///
/// The handler is otherwise free-standing, so this is all that stood between
/// it and being shared. `pounce-algorithm` implements it over its
/// `IpoptDataHandle`; `pounce-qp` (which cannot depend on `pounce-algorithm` —
/// the dependency runs the other way) passes `None` or its own sink.
pub trait PerturbationSink {
    /// Append to the iteration line's info string (upstream `info_string`).
    fn append_info(&self, s: &str);
    /// Record the primal regularization actually applied (upstream
    /// `info_regu_x`), for the `lg(rg)` column.
    fn set_regu_x(&self, v: Number);
    /// Current iteration index, for the env-gated `POUNCE_DBG_PERT` trace.
    fn iter_count(&self) -> Index {
        -1
    }
    /// Emit one debug line. Defaulted to a no-op so this crate stays
    /// dependency-light — `tracing` wiring belongs to the caller.
    fn debug(&self, _msg: &str) {}
}

/// Trial state — port of upstream `TrialStatus` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialStatus {
    NoTest,
    DcEq0DxEq0,
    DcGt0DxEq0,
    DcEq0DxGt0,
    DcGt0DxGt0,
}

/// Degeneracy state — port of `DegenType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegenType {
    NotDegenerate,
    Degenerate,
    NotYetDetermined,
}

/// State + algorithmic parameters. Defaults mirror
/// `IpPDPerturbationHandler.cpp::RegisterOptions`.
#[derive(Debug, Clone)]
pub struct PdPerturbationHandler {
    // ---- algorithmic parameters (read from options) ----
    pub delta_xs_max: Number,
    pub delta_xs_min: Number,
    pub delta_xs_first_inc_fact: Number,
    pub delta_xs_inc_fact: Number,
    pub delta_xs_dec_fact: Number,
    pub delta_xs_init: Number,
    pub delta_cd_val: Number,
    pub delta_cd_exp: Number,
    pub perturb_always_cd: bool,
    pub reset_last: bool,
    pub degen_iters_max: Index,

    // ---- live state ----
    pub delta_x_curr: Number,
    pub delta_s_curr: Number,
    pub delta_c_curr: Number,
    pub delta_d_curr: Number,
    pub delta_x_last: Number,
    pub delta_s_last: Number,
    pub delta_c_last: Number,
    pub delta_d_last: Number,
    pub get_deltas_for_wrong_inertia_called: bool,
    pub hess_degenerate: DegenType,
    pub jac_degenerate: DegenType,
    pub degen_iters: Index,
    pub test_status: TrialStatus,
    /// gh#592: how many rungs of the `δ_x` ladder have been taken for
    /// the current aug-system while `δ_c > 0`. Reset by
    /// [`Self::consider_new_system`].
    pub delta_c_rungs: Index,
    /// gh#592: `δ_c` has already been withdrawn once for this
    /// aug-system, so it must not be raised again before the next
    /// [`Self::consider_new_system`]. Without the latch a linear solver
    /// that keeps reporting `Singular` for the same reason would put it
    /// straight back and the walk-back would cycle.
    pub delta_c_abandoned: bool,
    /// gh#592: the rung at which `δ_c` is withdrawn — see
    /// [`Self::perturb_for_wrong_inertia`]. `0` disables the walk-back
    /// and restores the pre-#592 escalation exactly.
    pub delta_c_max_rungs: Index,
    /// gh#981: the `δ_x` rung at which `δ_c` was withdrawn. The factor at
    /// `(δ_x, δ_c)` reported `WrongInertia`, i.e. was nonsingular; a
    /// certified `Singular` without `δ_c` refutes the withdrawal and the
    /// ladder resumes from here.
    pub delta_x_at_withdrawal: Number,
    /// gh#981: `δ_c` was withdrawn and then reinstated for this
    /// aug-system. Latched so that the walk-back does not fire a second
    /// time; one withdrawal and one reinstatement per aug-system at most.
    pub delta_c_reinstated: bool,
    /// gh#981: `δ_c` has been withdrawn and no factorization without it
    /// has reported back yet. Only the *first* `δ_c`-free factor can
    /// refute the withdrawal: a dependent row shows up as a zero pivot
    /// at the very first rung, every time, whereas a zero pivot that
    /// appears at one rung of a long `δ_x` climb and not at its
    /// neighbours is round-off (measured on `pooling_rt2stp`, where
    /// honouring the later ones moved the exact leg to a different
    /// local optimum).
    pub withdrawal_untested: bool,
}

impl Default for PdPerturbationHandler {
    fn default() -> Self {
        Self {
            delta_xs_max: 1e20,
            delta_xs_min: 1e-20,
            delta_xs_first_inc_fact: 100.0,
            delta_xs_inc_fact: 8.0,
            delta_xs_dec_fact: 1.0 / 3.0,
            delta_xs_init: 1e-4,
            delta_cd_val: 1e-8,
            delta_cd_exp: 0.25,
            perturb_always_cd: false,
            reset_last: false,
            degen_iters_max: 3,
            delta_x_curr: 0.0,
            delta_s_curr: 0.0,
            delta_c_curr: 0.0,
            delta_d_curr: 0.0,
            delta_x_last: 0.0,
            delta_s_last: 0.0,
            delta_c_last: 0.0,
            delta_d_last: 0.0,
            get_deltas_for_wrong_inertia_called: false,
            hess_degenerate: DegenType::NotYetDetermined,
            jac_degenerate: DegenType::NotYetDetermined,
            degen_iters: 0,
            test_status: TrialStatus::NoTest,
            // Three rungs is above anything the models δ_c exists for
            // ever need: on eigena2 and eigenb2 (the gh#540 / gh#544
            // motivating cases) δ_c is followed by at most one rung
            // before the factor is accepted. See the doc comment on
            // `perturb_for_wrong_inertia`.
            delta_c_rungs: 0,
            delta_c_abandoned: false,
            delta_c_max_rungs: 3,
            delta_x_at_withdrawal: 0.0,
            delta_c_reinstated: false,
            withdrawal_untested: false,
        }
    }
}

/// Snapshot of the four perturbations after a state-machine call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deltas {
    pub delta_x: Number,
    pub delta_s: Number,
    pub delta_c: Number,
    pub delta_d: Number,
}

impl PdPerturbationHandler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure `perturb_always_cd_` and rebuild the initial `jac`
    /// state. Mirrors upstream's `InitializeImpl`.
    pub fn set_perturb_always_cd(&mut self, on: bool) {
        self.perturb_always_cd = on;
        self.jac_degenerate = if on {
            DegenType::NotDegenerate
        } else {
            DegenType::NotYetDetermined
        };
    }

    /// First call when starting a new aug-system. `mu` is the current
    /// barrier parameter (used by the `δ_cd` formula).
    /// Returns `None` if no suitable starting perturbation could be
    /// found (the caller bails).
    pub fn consider_new_system(
        &mut self,
        mu: Number,
        ip_data: Option<&dyn PerturbationSink>,
    ) -> Option<Deltas> {
        self.finalize_test(ip_data);

        // Bookkeeping: roll the previous trial's `_curr` values into
        // `_last` (matches upstream cpp:158-183).
        if self.reset_last {
            self.delta_x_last = self.delta_x_curr;
            self.delta_s_last = self.delta_s_curr;
            self.delta_c_last = self.delta_c_curr;
            self.delta_d_last = self.delta_d_curr;
        } else {
            if self.delta_x_curr > 0.0 {
                self.delta_x_last = self.delta_x_curr;
            }
            if self.delta_s_curr > 0.0 {
                self.delta_s_last = self.delta_s_curr;
            }
            if self.delta_c_curr > 0.0 {
                self.delta_c_last = self.delta_c_curr;
            }
            if self.delta_d_curr > 0.0 {
                self.delta_d_last = self.delta_d_curr;
            }
        }

        let undet = matches!(self.hess_degenerate, DegenType::NotYetDetermined)
            || matches!(self.jac_degenerate, DegenType::NotYetDetermined);
        self.test_status = if undet {
            if self.perturb_always_cd {
                TrialStatus::DcGt0DxEq0
            } else {
                TrialStatus::DcEq0DxEq0
            }
        } else {
            TrialStatus::NoTest
        };

        let mut delta_c = if matches!(self.jac_degenerate, DegenType::Degenerate) {
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            append_info(ip_data, "l");
            v
        } else if self.perturb_always_cd {
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            v
        } else {
            self.delta_c_curr = 0.0;
            0.0
        };
        let mut delta_d = delta_c;
        self.delta_d_curr = delta_d;

        let mut delta_x = 0.0;
        let mut delta_s = 0.0;

        if matches!(self.hess_degenerate, DegenType::Degenerate) {
            self.delta_x_curr = 0.0;
            self.delta_s_curr = 0.0;
            if !self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            ) {
                return None;
            }
        }

        self.delta_x_curr = delta_x;
        self.delta_s_curr = delta_s;
        self.delta_c_curr = delta_c;
        self.delta_d_curr = delta_d;
        set_info_regu_x(ip_data, delta_x);
        self.get_deltas_for_wrong_inertia_called = false;
        // gh#592: the walk-back is scoped to a single aug-system.
        self.delta_c_rungs = 0;
        self.delta_c_abandoned = false;
        self.delta_x_at_withdrawal = 0.0;
        self.delta_c_reinstated = false;
        self.withdrawal_untested = false;

        Some(Deltas {
            delta_x,
            delta_s,
            delta_c,
            delta_d,
        })
    }

    /// Escalation after `Singular` factorization status. Mirrors
    /// `PerturbForSingularity` (cpp:245-364).
    pub fn perturb_for_singular(
        &mut self,
        mu: Number,
        ip_data: Option<&dyn PerturbationSink>,
    ) -> Option<Deltas> {
        self.perturb_for_singular_with(mu, false, ip_data)
    }

    /// [`Self::perturb_for_singular`] with the linear solver's verdict on
    /// the report attached (gh#981). `certified` is true when the
    /// factorization *found* a zero pivot — a rank deficiency it measured
    /// — and false when `Singular` stands in for an inertia count the
    /// solver declined to trust (gh#540). The two differ only after a
    /// gh#592 walk-back: a certified `Singular` from the `δ_c`-free
    /// factor refutes the withdrawal, because the factor at that `δ_x`
    /// *with* `δ_c` was nonsingular and only `δ_c` changed, so the
    /// singularity is in the constraint block and no `δ_x` can remove it.
    /// `δ_c` is then put back and the ladder resumes from the withdrawal
    /// rung. An uncertified `Singular` is answered on the `δ_x` ladder as
    /// before — on gh#884's biactive MPCC every post-withdrawal `Singular`
    /// is of that kind, and reinstating `δ_c` there costs the certificate.
    pub fn perturb_for_singular_with(
        &mut self,
        mu: Number,
        certified: bool,
        ip_data: Option<&dyn PerturbationSink>,
    ) -> Option<Deltas> {
        let mut delta_x = 0.0;
        let mut delta_s = 0.0;
        let mut delta_c = 0.0;
        let mut delta_d = 0.0;

        // gh#592: `δ_c` has already been tried and withdrawn for this
        // aug-system (see `maybe_withdraw_delta_c`). A further
        // `Singular` is the same evidence that did not respond to it,
        // so answer it on the `δ_x` ladder rather than putting `δ_c`
        // straight back — which is what the arms below would do, and
        // would cycle.
        let first_without_delta_c = self.withdrawal_untested;
        self.withdrawal_untested = false;
        if self.delta_c_abandoned && certified && first_without_delta_c && !self.delta_c_reinstated
        {
            // gh#981: the withdrawal is refuted — see the doc comment.
            self.delta_c_abandoned = false;
            self.delta_c_reinstated = true;
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            self.delta_d_curr = v;
            // The factor at the withdrawal rung with δ_c on is already
            // known to be `WrongInertia`; resume the ladder from there so
            // the rungs below it are not climbed a second time.
            if self.delta_x_curr < self.delta_x_at_withdrawal {
                self.delta_x_curr = self.delta_x_at_withdrawal;
                self.delta_s_curr = self.delta_x_at_withdrawal;
            }
            self.test_status = TrialStatus::NoTest;
            append_info(ip_data, "W");
            if !self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            ) {
                return None;
            }
            set_info_regu_x(ip_data, self.delta_x_curr);
            return Some(Deltas {
                delta_x: self.delta_x_curr,
                delta_s: self.delta_s_curr,
                delta_c: self.delta_c_curr,
                delta_d: self.delta_d_curr,
            });
        }
        if self.delta_c_abandoned {
            self.test_status = TrialStatus::NoTest;
            if !self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            ) {
                return None;
            }
            set_info_regu_x(ip_data, self.delta_x_curr);
            return Some(Deltas {
                delta_x: self.delta_x_curr,
                delta_s: self.delta_s_curr,
                delta_c: self.delta_c_curr,
                delta_d: self.delta_d_curr,
            });
        }

        // Upstream's `TrialStatus` arms below assert that the degeneracy
        // probe is still in the state that named it — `δ_x == 0` for the
        // `DxEq0` statuses, and so on. Those are `DBG_ASSERT`s upstream,
        // i.e. assumptions rather than invariants: every real linear
        // solver can report `Singular` from *any* rung of the δ_x ladder
        // (MUMPS `INFO(1) = -10`, MA27 `IFLAG = 3`, and — since pounce
        // gh#540 — feral's inertia-trust floor), at which point the probe
        // has already been disturbed and its arm no longer applies.
        // Abandon the probe and fall through to the determined-state path,
        // which asserts nothing and does the right thing from wherever the
        // perturbations happen to be: raise δ_c if it is still zero,
        // otherwise take a δ_x step.
        let probe_intact = match self.test_status {
            TrialStatus::DcEq0DxEq0 => self.delta_x_curr == 0.0 && self.delta_c_curr == 0.0,
            TrialStatus::DcGt0DxEq0 => self.delta_x_curr == 0.0 && self.delta_c_curr > 0.0,
            TrialStatus::DcEq0DxGt0 => self.delta_x_curr > 0.0 && self.delta_c_curr == 0.0,
            TrialStatus::DcGt0DxGt0 | TrialStatus::NoTest => true,
        };
        if !probe_intact {
            self.test_status = TrialStatus::NoTest;
        }

        let undet = probe_intact
            && (matches!(self.hess_degenerate, DegenType::NotYetDetermined)
                || matches!(self.jac_degenerate, DegenType::NotYetDetermined));
        if undet {
            match self.test_status {
                TrialStatus::DcEq0DxEq0 => {
                    debug_assert!(self.delta_x_curr == 0.0 && self.delta_c_curr == 0.0);
                    if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                        let v = self.delta_cd(mu);
                        self.delta_c_curr = v;
                        self.delta_d_curr = v;
                        self.test_status = TrialStatus::DcGt0DxEq0;
                    } else {
                        debug_assert!(matches!(self.hess_degenerate, DegenType::NotYetDetermined));
                        if !self.get_deltas_for_wrong_inertia(
                            &mut delta_x,
                            &mut delta_s,
                            &mut delta_c,
                            &mut delta_d,
                            ip_data,
                        ) {
                            return None;
                        }
                        self.test_status = TrialStatus::DcEq0DxGt0;
                    }
                }
                TrialStatus::DcGt0DxEq0 => {
                    debug_assert!(self.delta_x_curr == 0.0 && self.delta_c_curr > 0.0);
                    debug_assert!(matches!(self.jac_degenerate, DegenType::NotYetDetermined));
                    if !self.perturb_always_cd {
                        self.delta_c_curr = 0.0;
                        self.delta_d_curr = 0.0;
                        if !self.get_deltas_for_wrong_inertia(
                            &mut delta_x,
                            &mut delta_s,
                            &mut delta_c,
                            &mut delta_d,
                            ip_data,
                        ) {
                            return None;
                        }
                        self.test_status = TrialStatus::DcEq0DxGt0;
                    } else if !self.get_deltas_for_wrong_inertia(
                        &mut delta_x,
                        &mut delta_s,
                        &mut delta_c,
                        &mut delta_d,
                        ip_data,
                    ) {
                        return None;
                    } else {
                        self.test_status = TrialStatus::DcGt0DxGt0;
                    }
                }
                TrialStatus::DcEq0DxGt0 => {
                    debug_assert!(self.delta_x_curr > 0.0 && self.delta_c_curr == 0.0);
                    let v = self.delta_cd(mu);
                    self.delta_c_curr = v;
                    self.delta_d_curr = v;
                    if !self.get_deltas_for_wrong_inertia(
                        &mut delta_x,
                        &mut delta_s,
                        &mut delta_c,
                        &mut delta_d,
                        ip_data,
                    ) {
                        return None;
                    }
                    self.test_status = TrialStatus::DcGt0DxGt0;
                }
                TrialStatus::DcGt0DxGt0 => {
                    if !self.get_deltas_for_wrong_inertia(
                        &mut delta_x,
                        &mut delta_s,
                        &mut delta_c,
                        &mut delta_d,
                        ip_data,
                    ) {
                        return None;
                    }
                }
                TrialStatus::NoTest => {
                    debug_assert!(false, "perturb_for_singular: NoTest in undetermined branch");
                }
            }
        } else if self.delta_c_curr > 0.0 {
            // Already perturbed C; treat as wrong-inertia.
            if !self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            ) {
                return None;
            }
        } else {
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            self.delta_d_curr = v;
            append_info(ip_data, "L");
        }

        let out = Deltas {
            delta_x: self.delta_x_curr,
            delta_s: self.delta_s_curr,
            delta_c: self.delta_c_curr,
            delta_d: self.delta_d_curr,
        };
        set_info_regu_x(ip_data, out.delta_x);
        Some(out)
    }

    /// Escalation after `WrongInertia` factorization status. Mirrors
    /// `PerturbForWrongInertia` (cpp:419-450).
    pub fn perturb_for_wrong_inertia(
        &mut self,
        mu: Number,
        ip_data: Option<&dyn PerturbationSink>,
    ) -> Option<Deltas> {
        if std::env::var_os("POUNCE_DBG_PERT").is_some() {
            if let Some(d) = ip_data {
                d.debug(&format!(
                    "[PERT] iter={} WRONG_INERTIA mu={:.2e} dx_last={:.2e} dx_curr={:.2e}",
                    d.iter_count(),
                    mu,
                    self.delta_x_last,
                    self.delta_x_curr
                ));
            }
        }
        self.finalize_test(ip_data);
        // gh#981: a `WrongInertia` from the first `δ_c`-free factor is
        // gh#592's full-rank case; the withdrawal stands.
        self.withdrawal_untested = false;
        self.maybe_withdraw_delta_c(ip_data);

        let mut delta_x = 0.0;
        let mut delta_s = 0.0;
        let mut delta_c = 0.0;
        let mut delta_d = 0.0;
        let mut ok = self.get_deltas_for_wrong_inertia(
            &mut delta_x,
            &mut delta_s,
            &mut delta_c,
            &mut delta_d,
            ip_data,
        );
        // Upstream "no progress on δ_x but δ_c == 0" recovery: bring
        // up the C/D perturbation, reset Hessian degeneracy, retry.
        // Upstream peeks at the OUT-parameter `delta_c`, but
        // `get_deltas_for_wrong_inertia` only writes that on success;
        // we look at the handler's own δ_c_curr instead, which
        // matches the algorithmic intent unambiguously.
        if !ok && self.delta_c_curr == 0.0 {
            debug_assert_eq!(self.delta_d_curr, 0.0);
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            self.delta_d_curr = v;
            self.delta_x_curr = 0.0;
            self.delta_s_curr = 0.0;
            self.test_status = TrialStatus::NoTest;
            if matches!(self.hess_degenerate, DegenType::Degenerate) {
                self.hess_degenerate = DegenType::NotYetDetermined;
            }
            ok = self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            );
        }
        if !ok {
            return None;
        }
        Some(Deltas {
            delta_x,
            delta_s,
            delta_c,
            delta_d,
        })
    }

    /// gh#592: withdraw `δ_c` once it has demonstrably failed to buy a
    /// usable inertia.
    ///
    /// `δ_c` is the perturbation for a rank-deficient constraint
    /// Jacobian, and it is reached for when the factorization reports
    /// `Singular`. Since gh#540 a factorization also reports `Singular`
    /// when its inertia is *unmeasurable* — the count disagrees and the
    /// smallest pivot is at the noise floor — which is evidence about
    /// the measurement, not about the Jacobian's rank. When the
    /// Jacobian in fact has full rank, `δ_c` cannot help, and because it
    /// stays switched on for the rest of the aug-system the `δ_x` ladder
    /// then has to climb against a matrix `δ_c` has made *harder* to hit
    /// the requested inertia on: on the gh#592 model this cost five
    /// rungs, ending at `δ_w = 1e2` where Ipopt accepted the step at
    /// `1e-4`, and the over-damped step froze the objective for the next
    /// eight iterations before the loose-tolerance exit test fired.
    ///
    /// So rather than predict which kind of `Singular` this was — the
    /// counts are the very thing gh#540 established are noise — let the
    /// ladder answer it. After `delta_c_max_rungs` rungs with `δ_c` on
    /// and still no acceptable inertia, `δ_c` has had its chance:
    /// withdraw it, restart the `δ_x` ladder, and latch it off for the
    /// remainder of this aug-system. Where `δ_c` is the right remedy
    /// this never fires — on eigena2 and eigenb2 it is followed by at
    /// most one rung.
    fn maybe_withdraw_delta_c(&mut self, ip_data: Option<&dyn PerturbationSink>) {
        if self.delta_c_max_rungs <= 0
            || self.delta_c_abandoned
            || self.delta_c_reinstated
            || self.delta_c_curr <= 0.0
        {
            return;
        }
        self.delta_c_rungs += 1;
        if self.delta_c_rungs < self.delta_c_max_rungs {
            return;
        }
        self.delta_x_at_withdrawal = self.delta_x_curr;
        self.withdrawal_untested = true;
        self.delta_c_curr = 0.0;
        self.delta_d_curr = 0.0;
        self.delta_x_curr = 0.0;
        self.delta_s_curr = 0.0;
        self.delta_c_abandoned = true;
        self.test_status = TrialStatus::NoTest;
        append_info(ip_data, "w");
    }

    /// Read the most recently committed perturbations.
    pub fn current_perturbation(&self) -> Deltas {
        Deltas {
            delta_x: self.delta_x_curr,
            delta_s: self.delta_s_curr,
            delta_c: self.delta_c_curr,
            delta_d: self.delta_d_curr,
        }
    }

    /// Internal — pure escalation of `δ_x` / `δ_s`. Returns `false` if
    /// `δ_x` would exceed `delta_xs_max`. Mirrors
    /// `get_deltas_for_wrong_inertia`.
    fn get_deltas_for_wrong_inertia(
        &mut self,
        delta_x: &mut Number,
        delta_s: &mut Number,
        delta_c: &mut Number,
        delta_d: &mut Number,
        ip_data: Option<&dyn PerturbationSink>,
    ) -> bool {
        if self.delta_x_curr == 0.0 {
            self.delta_x_curr = if self.delta_x_last == 0.0 {
                self.delta_xs_init
            } else {
                self.delta_xs_min
                    .max(self.delta_x_last * self.delta_xs_dec_fact)
            };
        } else if self.delta_x_last == 0.0 || 1e5 * self.delta_x_last < self.delta_x_curr {
            self.delta_x_curr *= self.delta_xs_first_inc_fact;
        } else {
            self.delta_x_curr *= self.delta_xs_inc_fact;
        }
        if self.delta_x_curr > self.delta_xs_max {
            self.delta_x_last = 0.0;
            self.delta_s_last = 0.0;
            append_info(ip_data, "dx");
            return false;
        }
        self.delta_s_curr = self.delta_x_curr;

        *delta_x = self.delta_x_curr;
        *delta_s = self.delta_s_curr;
        *delta_c = self.delta_c_curr;
        *delta_d = self.delta_d_curr;
        set_info_regu_x(ip_data, *delta_x);
        self.get_deltas_for_wrong_inertia_called = true;
        true
    }

    fn delta_cd(&self, mu: Number) -> Number {
        self.delta_cd_val * mu.powf(self.delta_cd_exp)
    }

    /// Read the test outcome from the just-completed (non-singular)
    /// factor and update degeneracy flags. Mirrors `finalize_test`
    /// (cpp:470-538).
    fn finalize_test(&mut self, ip_data: Option<&dyn PerturbationSink>) {
        match self.test_status {
            TrialStatus::NoTest => (),
            TrialStatus::DcEq0DxEq0 => {
                if matches!(self.hess_degenerate, DegenType::NotYetDetermined)
                    && matches!(self.jac_degenerate, DegenType::NotYetDetermined)
                {
                    self.hess_degenerate = DegenType::NotDegenerate;
                    self.jac_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nhj ");
                } else if matches!(self.hess_degenerate, DegenType::NotYetDetermined) {
                    self.hess_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nh ");
                } else if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                    self.jac_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nj ");
                }
            }
            TrialStatus::DcGt0DxEq0 => {
                if matches!(self.hess_degenerate, DegenType::NotYetDetermined) {
                    self.hess_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nh ");
                }
                if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                    self.degen_iters += 1;
                    if self.degen_iters >= self.degen_iters_max {
                        self.jac_degenerate = DegenType::Degenerate;
                        append_info(ip_data, "Dj ");
                    }
                    append_info(ip_data, "L");
                }
            }
            TrialStatus::DcEq0DxGt0 => {
                if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                    self.jac_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nj ");
                }
                if matches!(self.hess_degenerate, DegenType::NotYetDetermined) {
                    self.degen_iters += 1;
                    if self.degen_iters >= self.degen_iters_max {
                        self.hess_degenerate = DegenType::Degenerate;
                        append_info(ip_data, "Dh ");
                    }
                }
            }
            TrialStatus::DcGt0DxGt0 => {
                self.degen_iters += 1;
                if self.degen_iters >= self.degen_iters_max {
                    self.hess_degenerate = DegenType::Degenerate;
                    self.jac_degenerate = DegenType::Degenerate;
                    append_info(ip_data, "Dhj ");
                }
                append_info(ip_data, "L");
            }
        }
    }
}

fn append_info(sink: Option<&dyn PerturbationSink>, s: &str) {
    if let Some(h) = sink {
        h.append_info(s);
    }
}

fn set_info_regu_x(sink: Option<&dyn PerturbationSink>, v: Number) {
    if let Some(h) = sink {
        h.set_regu_x(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#592: three rungs of the `δ_x` ladder with `δ_c` on and still
    /// no acceptable inertia is `δ_c` failing at the job it was raised
    /// for, so it is withdrawn and the ladder restarts without it.
    #[test]
    fn delta_c_is_withdrawn_after_the_ladder_has_climbed_without_it() {
        let mut h = PdPerturbationHandler::new();
        h.consider_new_system(0.1, None).unwrap();

        // A `Singular` factor raises δ_c on its own — no δ_x yet.
        let d = h.perturb_for_singular(0.1, None).unwrap();
        assert!(d.delta_c > 0.0, "δ_c should be the first response");
        assert_eq!(d.delta_x, 0.0);

        // Rungs one and two: δ_c stays, because it may yet pay off.
        for rung in 1..=2 {
            let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
            assert!(d.delta_c > 0.0, "δ_c withdrawn early, at rung {rung}");
            assert!(d.delta_x > 0.0);
        }

        // Rung three: withdrawn, and the ladder restarts from the low
        // rung rather than carrying the height it reached under δ_c.
        let before = h.delta_x_curr;
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert_eq!(d.delta_c, 0.0, "δ_c was not withdrawn at the third rung");
        assert_eq!(d.delta_d, 0.0);
        assert!(
            d.delta_x > 0.0 && d.delta_x < before,
            "the ladder did not restart"
        );
    }

    /// ...and it stays withdrawn for the rest of this aug-system. The
    /// factorization that reported `Singular` will keep reporting it for
    /// the same reason, so without the latch δ_c would go straight back
    /// on and the walk-back would cycle.
    #[test]
    fn a_withdrawn_delta_c_is_not_raised_again_until_the_next_iterate() {
        let mut h = PdPerturbationHandler::new();
        h.consider_new_system(0.1, None).unwrap();
        h.perturb_for_singular(0.1, None).unwrap();
        for _ in 0..3 {
            h.perturb_for_wrong_inertia(0.1, None).unwrap();
        }
        assert!(h.delta_c_abandoned);

        let d = h.perturb_for_singular(0.1, None).unwrap();
        assert_eq!(d.delta_c, 0.0, "δ_c came back inside the same aug-system");
        assert!(d.delta_x > 0.0, "the `Singular` was not answered at all");

        // The next iterate starts clean: the withdrawal is a statement
        // about one aug-system, not about the problem.
        h.consider_new_system(0.1, None).unwrap();
        assert!(!h.delta_c_abandoned);
        assert_eq!(h.delta_c_rungs, 0);
        let d = h.perturb_for_singular(0.1, None).unwrap();
        assert!(d.delta_c > 0.0, "δ_c is still latched off a new aug-system");
    }

    /// Where `δ_c` is the right remedy the walk-back must be invisible.
    /// One rung is the whole pattern on eigena2 and eigenb2, the models
    /// gh#540 / gh#544 raised δ_c for.
    #[test]
    fn one_rung_under_delta_c_leaves_it_alone() {
        let mut h = PdPerturbationHandler::new();
        h.consider_new_system(0.1, None).unwrap();
        h.perturb_for_singular(0.1, None).unwrap();
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!(d.delta_c > 0.0);
        // A good factor ends the aug-system; the counter resets with it.
        h.consider_new_system(0.1, None).unwrap();
        assert_eq!(h.delta_c_rungs, 0);
    }

    /// The opt-out restores the pre-#592 escalation exactly: δ_c on,
    /// and the δ_x ladder climbing against it without bound.
    #[test]
    fn zero_max_rungs_disables_the_walkback() {
        let mut h = PdPerturbationHandler::new();
        h.delta_c_max_rungs = 0;
        h.consider_new_system(0.1, None).unwrap();
        h.perturb_for_singular(0.1, None).unwrap();
        let mut last = 0.0;
        for _ in 0..6 {
            let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
            assert!(d.delta_c > 0.0, "δ_c was withdrawn with the walk-back off");
            assert!(d.delta_x > last, "the ladder stopped climbing");
            last = d.delta_x;
        }
        assert!(!h.delta_c_abandoned);
    }

    #[test]
    fn first_wrong_inertia_perturbation_is_delta_xs_init() {
        let mut h = PdPerturbationHandler::new();
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        // delta_xs_init = first_hessian_perturbation = 1e-4
        assert!((d.delta_x - 1e-4).abs() < 1e-20);
        assert_eq!(d.delta_x, d.delta_s);
        assert_eq!(d.delta_c, 0.0);
        assert_eq!(d.delta_d, 0.0);
    }

    #[test]
    fn second_perturbation_uses_first_inc_fact() {
        // After the *first* nonzero δ_x, with δ_x_last == 0, the
        // doubling uses `delta_xs_first_inc_fact = 100` per upstream
        // (cpp:386-389: "if delta_x_last_ == 0 ...").
        let mut h = PdPerturbationHandler::new();
        let d1 = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        let d2 = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!((d2.delta_x - d1.delta_x * 100.0).abs() < 1e-15);
    }

    #[test]
    fn third_perturbation_uses_inc_fact() {
        // After delta_x_last has been set (via consider_new_system or
        // first inc), continued growth uses `delta_xs_inc_fact = 8`.
        let mut h = PdPerturbationHandler::new();
        h.delta_x_curr = 1e-2;
        h.delta_x_last = 1e-2;
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!((d.delta_x - 1e-2 * 8.0).abs() < 1e-15);
    }

    #[test]
    fn perturbation_caps_at_max_when_dcd_already_active() {
        // When δ_c is already > 0 (e.g., perturb_always_cd, or after a
        // singular-recovery), the fallback path inside
        // `perturb_for_wrong_inertia` is skipped and the
        // δ_x-overflow surfaces as `None`.
        let mut h = PdPerturbationHandler::new();
        h.delta_x_curr = h.delta_xs_max;
        h.delta_c_curr = 1e-4;
        h.delta_d_curr = 1e-4;
        assert!(h.perturb_for_wrong_inertia(0.1, None).is_none());
    }

    #[test]
    fn consider_new_system_with_perturb_always_cd_seeds_dcd() {
        let mut h = PdPerturbationHandler::new();
        h.set_perturb_always_cd(true);
        let mu = 0.1;
        let d = h.consider_new_system(mu, None).unwrap();
        let expected = h.delta_cd_val * mu.powf(h.delta_cd_exp);
        assert!((d.delta_c - expected).abs() < 1e-15);
        assert!((d.delta_d - expected).abs() < 1e-15);
        assert_eq!(d.delta_x, 0.0);
        assert_eq!(d.delta_s, 0.0);
    }

    #[test]
    fn consider_new_system_default_zeros_dcd() {
        let mut h = PdPerturbationHandler::new();
        let d = h.consider_new_system(0.1, None).unwrap();
        assert_eq!(
            d,
            Deltas {
                delta_x: 0.0,
                delta_s: 0.0,
                delta_c: 0.0,
                delta_d: 0.0
            }
        );
    }

    #[test]
    fn singular_in_test_dc_eq0_dx_eq0_seeds_dcd() {
        let mut h = PdPerturbationHandler::new();
        let _ = h.consider_new_system(0.1, None).unwrap();
        // After consider_new_system on a fresh handler, test_status
        // should be DcEq0DxEq0 (since both flags are NotYetDetermined,
        // and perturb_always_cd is false).
        assert_eq!(h.test_status, TrialStatus::DcEq0DxEq0);
        let d = h.perturb_for_singular(0.1, None).unwrap();
        let expected = h.delta_cd_val * (0.1_f64).powf(h.delta_cd_exp);
        assert!((d.delta_c - expected).abs() < 1e-15);
        assert!((d.delta_d - expected).abs() < 1e-15);
        assert_eq!(d.delta_x, 0.0);
        assert_eq!(h.test_status, TrialStatus::DcGt0DxEq0);
    }

    #[test]
    fn singular_when_determined_with_dc_zero_seeds_dcd() {
        let mut h = PdPerturbationHandler::new();
        h.hess_degenerate = DegenType::NotDegenerate;
        h.jac_degenerate = DegenType::NotDegenerate;
        h.test_status = TrialStatus::NoTest;
        let d = h.perturb_for_singular(0.1, None).unwrap();
        let expected = h.delta_cd_val * (0.1_f64).powf(h.delta_cd_exp);
        assert!((d.delta_c - expected).abs() < 1e-15);
    }

    /// gh#540: the case upstream's `DBG_ASSERT`s actually trip on. Reach
    /// `perturb_for_singular` a *second* time, from a rung of the δ_x ladder,
    /// while the Jacobian flag is still undetermined — so `finalize_test` has
    /// not yet resolved it and the `DcGt0DxEq0` arm is entered with
    /// `δ_x > 0`, against its `debug_assert!(delta_x_curr == 0.0 && ...)`.
    /// Every real linear solver can produce this sequence (MUMPS
    /// `INFO(1) = -10`, MA27 `IFLAG = 3`, and pounce's inertia-trust floor
    /// all report singularity from anywhere on the ladder), so the
    /// precondition is an assumption rather than an invariant, and the
    /// handler has to survive it rather than assert it.
    #[test]
    fn a_second_singular_verdict_from_the_ladder_does_not_trip_the_probe() {
        let mut h = PdPerturbationHandler::new();
        let _ = h.consider_new_system(0.1, None).unwrap();
        // Singular at δ_x = 0 → the probe raises δ_c and moves to DcGt0DxEq0,
        // leaving the Jacobian flag undetermined.
        let _ = h.perturb_for_singular(0.1, None).unwrap();
        assert_eq!(h.test_status, TrialStatus::DcGt0DxEq0);
        assert_eq!(h.jac_degenerate, DegenType::NotYetDetermined);
        // WrongInertia next → δ_x leaves zero. `finalize_test` resolves the
        // Hessian flag but leaves the Jacobian one undetermined (it needs
        // `degen_iters_max` trials), so the probe is still nominally running.
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!(d.delta_x > 0.0);
        assert_eq!(h.jac_degenerate, DegenType::NotYetDetermined);
        assert_eq!(h.test_status, TrialStatus::DcGt0DxEq0);
        // ...and now Singular again, with δ_x > 0 under a `DxEq0` status.
        // Pre-#540 this is the `debug_assert` that fires.
        let d = h
            .perturb_for_singular(0.1, None)
            .expect("a singular verdict from the ladder must not be fatal");
        assert!(
            d.delta_c > 0.0,
            "δ_c was dropped by the abandoned probe: {}",
            d.delta_c,
        );
        assert_eq!(
            h.test_status,
            TrialStatus::NoTest,
            "a probe whose precondition no longer holds must be abandoned",
        );
    }

    /// The same guard on the commoner sequence, where `finalize_test` has
    /// already resolved both flags by the time the `Singular` verdict lands:
    /// δ_c comes up and the δ_x rung the ladder already paid for is kept.
    #[test]
    fn singular_after_a_delta_x_step_abandons_the_probe() {
        let mut h = PdPerturbationHandler::new();
        // Fresh system: both flags undetermined, so the probe arms as
        // DcEq0DxEq0 with δ_x = δ_c = 0.
        let _ = h.consider_new_system(0.1, None).unwrap();
        assert_eq!(h.test_status, TrialStatus::DcEq0DxEq0);
        // First factorization came back WrongInertia — δ_x leaves zero while
        // `test_status` still says the probe is running at δ_x = 0.
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!(d.delta_x > 0.0);
        // Second factorization comes back Singular.
        let d = h.perturb_for_singular(0.1, None).unwrap();
        let expected_cd = h.delta_cd_val * (0.1_f64).powf(h.delta_cd_exp);
        assert!(
            (d.delta_c - expected_cd).abs() < 1e-20,
            "δ_c was not raised: {}",
            d.delta_c,
        );
        assert_eq!(d.delta_c, d.delta_d);
        assert_eq!(
            d.delta_x, h.delta_xs_init,
            "the δ_x ladder lost the rung it had already paid for",
        );
        assert_eq!(
            h.test_status,
            TrialStatus::NoTest,
            "a probe whose precondition no longer holds must be abandoned, \
             not carried into finalize_test",
        );
    }

    #[test]
    fn finalize_test_sets_not_degenerate_after_dc_eq0_dx_eq0_pass() {
        let mut h = PdPerturbationHandler::new();
        let _ = h.consider_new_system(0.1, None).unwrap();
        // Simulate the next call recognizing that the previous trial
        // factor was non-singular: a fresh consider_new_system runs
        // finalize_test first.
        let _ = h.consider_new_system(0.1, None).unwrap();
        assert_eq!(h.hess_degenerate, DegenType::NotDegenerate);
        assert_eq!(h.jac_degenerate, DegenType::NotDegenerate);
    }

    #[test]
    fn current_perturbation_returns_committed_values() {
        let mut h = PdPerturbationHandler::new();
        h.delta_x_curr = 1.0;
        h.delta_s_curr = 2.0;
        h.delta_c_curr = 3.0;
        h.delta_d_curr = 4.0;
        let d = h.current_perturbation();
        assert_eq!(d.delta_x, 1.0);
        assert_eq!(d.delta_s, 2.0);
        assert_eq!(d.delta_c, 3.0);
        assert_eq!(d.delta_d, 4.0);
    }

    /// Drive the handler to a gh#592 withdrawal: `δ_c` up on a `Singular`,
    /// then three `WrongInertia` rungs under it. Returns the rung `δ_c`
    /// was withdrawn at.
    fn withdraw(h: &mut PdPerturbationHandler) -> Number {
        h.consider_new_system(0.1, None).unwrap();
        h.perturb_for_singular(0.1, None).unwrap();
        let mut at = 0.0;
        for _ in 0..3 {
            at = h.delta_x_curr;
            h.perturb_for_wrong_inertia(0.1, None).unwrap();
        }
        assert!(h.delta_c_abandoned, "precondition: δ_c was withdrawn");
        assert_eq!(h.delta_x_at_withdrawal, at);
        at
    }

    /// gh#981: the first factorization without `δ_c` comes back with a
    /// zero pivot the solver found. Only `δ_c` changed between that
    /// factor and the nonsingular one at the withdrawal rung, so the
    /// Jacobian is rank-deficient and `δ_c` goes back — with the ladder
    /// resumed from the withdrawal rung, not restarted below it. On the
    /// gh#981 CSTR this is the difference between a peak `δ_w` of
    /// `2.1e-3` and one of `1.09`.
    #[test]
    fn a_certified_singular_from_the_first_delta_c_free_factor_reinstates_delta_c() {
        let mut h = PdPerturbationHandler::new();
        let at = withdraw(&mut h);
        let d = h.perturb_for_singular_with(0.1, true, None).unwrap();
        assert!(d.delta_c > 0.0, "δ_c was not reinstated");
        assert_eq!(d.delta_d, d.delta_c);
        assert!(
            d.delta_x > at,
            "the ladder restarted below the withdrawal rung ({} <= {at})",
            d.delta_x
        );
        assert!(h.delta_c_reinstated && !h.delta_c_abandoned);

        // And it stays: the walk-back does not fire a second time in this
        // aug-system, however many rungs follow.
        for _ in 0..6 {
            let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
            assert!(d.delta_c > 0.0, "δ_c withdrawn a second time");
        }
        // A new aug-system starts clean.
        h.consider_new_system(0.1, None).unwrap();
        assert!(!h.delta_c_reinstated && !h.withdrawal_untested);
    }

    /// gh#981: an *uncertified* `Singular` — gh#540's untrusted-inertia
    /// report — is the evidence the walk-back exists for, and does not
    /// reinstate. This is the shape of every post-withdrawal `Singular`
    /// on gh#884's `mpcc_qpec_small_biactive`, where reinstating costs the
    /// certificate.
    #[test]
    fn an_uncertified_singular_does_not_reinstate_delta_c() {
        let mut h = PdPerturbationHandler::new();
        withdraw(&mut h);
        for _ in 0..5 {
            let d = h.perturb_for_singular_with(0.1, false, None).unwrap();
            assert_eq!(d.delta_c, 0.0, "δ_c came back on an untrusted report");
            assert!(d.delta_x > 0.0);
        }
        assert!(!h.delta_c_reinstated);
    }

    /// gh#981: only the *first* `δ_c`-free factor can refute the
    /// withdrawal. A zero pivot that appears several rungs up a climb
    /// whose first rungs were nonsingular is round-off, not a dependent
    /// row — measured on `pooling_rt2stp`, where honouring it moved the
    /// exact leg onto a different local optimum.
    #[test]
    fn a_certified_singular_later_in_the_climb_does_not_reinstate_delta_c() {
        let mut h = PdPerturbationHandler::new();
        withdraw(&mut h);
        // First δ_c-free factor: `WrongInertia` — gh#592's full-rank case.
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert_eq!(d.delta_c, 0.0);
        assert!(!h.withdrawal_untested);
        // A certified `Singular` two rungs later is too late.
        h.perturb_for_singular_with(0.1, false, None).unwrap();
        let d = h.perturb_for_singular_with(0.1, true, None).unwrap();
        assert_eq!(d.delta_c, 0.0, "a late zero pivot reinstated δ_c");
        assert!(!h.delta_c_reinstated);
    }

    /// gh#981: an uncertified first report also closes the window, so a
    /// certified one right after it does not reinstate either.
    #[test]
    fn an_uncertified_first_report_closes_the_window() {
        let mut h = PdPerturbationHandler::new();
        withdraw(&mut h);
        h.perturb_for_singular_with(0.1, false, None).unwrap();
        let d = h.perturb_for_singular_with(0.1, true, None).unwrap();
        assert_eq!(d.delta_c, 0.0);
    }

    /// gh#981: before any withdrawal a certified `Singular` is the
    /// ordinary `perturb_for_singular` path — `δ_c` up, nothing latched.
    #[test]
    fn a_certified_singular_before_any_withdrawal_is_the_ordinary_path() {
        let mut h = PdPerturbationHandler::new();
        h.consider_new_system(0.1, None).unwrap();
        let a = h.perturb_for_singular_with(0.1, true, None).unwrap();
        let mut g = PdPerturbationHandler::new();
        g.consider_new_system(0.1, None).unwrap();
        let b = g.perturb_for_singular(0.1, None).unwrap();
        assert_eq!(a, b);
        assert!(!h.delta_c_reinstated && !h.delta_c_abandoned);
    }
}
