//! The matrix-form convex-QP pipeline a frontend wraps around an engine:
//! the engine-switch screen, the PSD refusal on the interior-point engine,
//! presolve → solve → postsolve, and the post-presolve demotion of an
//! unverified `Optimal`.
//!
//! Lifted out of the Python bindings (`pounce-py/src/qp.rs`, gh#1147 S3-I1)
//! so that every frontend which takes a `QpProblem` in matrix form — the
//! Python `solve_qp` today, the C API's planned `PounceQp*` handle next —
//! runs one implementation rather than two copies that can drift. Nothing here changes
//! what any solve does: the functions are the bindings' own, moved.
//!
//! This is **not** the CLI's convex dispatch. `pounce-cli` reaches the convex
//! engines through `run_convex_qp` on a model read from a `.nl` file, which
//! carries its own presolve and demotion; [`solve_presolved`] and
//! [`demote_unverified_optimal`] reproduce that sequence minus the `.nl`
//! plumbing, and are what a caller that already holds a `QpProblem` uses.

use crate::ipm::QpOptions;
use crate::presolve::{PresolveOutcome, PresolveStats, presolve};
use crate::qp::{QpProblem, QpSolution, QpStatus};
use pounce_qp::HessianInertia;
use std::fmt;

/// A refusal from the pipeline's screens ([`EngineSwitches::check`],
/// [`check_ipm_inertia`]): a combination of inputs that would otherwise be
/// accepted and silently dropped, or answered wrongly.
///
/// `Display` is the reason alone, without the name of the entry point; a
/// frontend prefixes its own (Python's `solve_qp` writes `"solve_qp: "`).
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    /// `qp_reg` is negative or non-finite.
    InvalidReg(f64),
    /// An interior-point-only switch was set for the active-set engine.
    IpmOnlyOption(&'static str),
    /// `qp_presolve = true` together with a warm start.
    PresolveWithWarmStart,
    /// `qp_crossover = true` together with a warm start.
    CrossoverWithWarmStart,
    /// The caller reports an indefinite `P` and asked for the interior-point
    /// engine (gh #112).
    IndefiniteUnderIpm,
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::InvalidReg(r) => write!(
                f,
                "`qp_reg` must be a finite, non-negative number (the static \
                 KKT regularization δ); got {r}"
            ),
            PipelineError::IpmOnlyOption(name) => write!(
                f,
                "`{name}` is an option of the convex interior-point engine \
                 (method='ipm'); the active-set engine has no such setting, so it is \
                 refused rather than ignored"
            ),
            PipelineError::PresolveWithWarmStart => f.write_str(
                "qp_presolve=True cannot be combined with warm_start=: the \
                 warm point is in the original variables and presolve solves a reduced \
                 problem, so the start would be discarded",
            ),
            PipelineError::CrossoverWithWarmStart => f.write_str(
                "qp_crossover=True cannot be combined with warm_start=: the \
                 warm-start path runs no crossover",
            ),
            PipelineError::IndefiniteUnderIpm => f.write_str(
                "the convex interior-point engine (method='ipm') requires a \
                 positive-semidefinite P and reports a silently-wrong 'optimal' at a \
                 saddle point without one; pass method='active-set' to solve an \
                 indefinite QP",
            ),
        }
    }
}

impl std::error::Error for PipelineError {}

/// The CLI's convex-engine switches (`qp_presolve`, `qp_reg`, `qp_hsde`,
/// `qp_equilibrate`, `qp_crossover`) as a matrix-form entry point takes them
/// (gh#990; asked for by discopt#1615).
///
/// They were CLI-only until Python's `solve_qp` took them: a caller had no way
/// to ask for a presolve, an exact LP vertex, or the direct driver. `None`
/// leaves each at what the entry point did before, which is the [`QpOptions`]
/// default for the last four and **no presolve** — the CLI presolves by
/// default, but the matrix-form path never has, and an omitted argument must
/// not change a solve.
///
/// What each engine reads, and therefore what is refused rather than accepted
/// and dropped:
///
/// * the interior-point engine (`method="ipm"`) reads all five.
/// * the active-set engine (`method="active-set"`) reads `presolve` (the CLI
///   presolves for both engines) and `equilibrate` (its retry on a
///   badly-scaled problem). It has no KKT regularization, no embedding, and
///   already returns a vertex, so `reg`, `hsde` and `crossover` are refused
///   there.
/// * With a warm start, the IPM's warm path runs no crossover and the warm
///   point lives in the unreduced space, so `crossover = Some(true)` and
///   `presolve = Some(true)` are refused. (`Some(false)` asks for what happens
///   anyway.)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineSwitches {
    pub presolve: Option<bool>,
    pub reg: Option<f64>,
    pub hsde: Option<bool>,
    pub equilibrate: Option<bool>,
    pub crossover: Option<bool>,
}

impl EngineSwitches {
    /// Refuse a switch the selected engine would drop. `active_set` is true
    /// when the active-set engine is selected (any other selection is screened
    /// as the interior-point engine; an unknown engine name is the caller's to
    /// reject, after this check, as Python's `solve_qp` does); `warm` is true
    /// when a warm start was supplied. The checks run in a fixed order and the
    /// first failure is returned.
    pub fn check(&self, active_set: bool, warm: bool) -> Result<(), PipelineError> {
        if let Some(r) = self.reg
            && (!r.is_finite() || r < 0.0)
        {
            return Err(PipelineError::InvalidReg(r));
        }
        if active_set {
            let ipm_only = [
                ("qp_reg", self.reg.is_some()),
                ("qp_hsde", self.hsde.is_some()),
                ("qp_crossover", self.crossover.is_some()),
            ];
            if let Some((name, _)) = ipm_only.iter().find(|(_, set)| *set) {
                return Err(PipelineError::IpmOnlyOption(name));
            }
        }
        if warm {
            if self.presolve == Some(true) {
                return Err(PipelineError::PresolveWithWarmStart);
            }
            if self.crossover == Some(true) {
                return Err(PipelineError::CrossoverWithWarmStart);
            }
        }
        Ok(())
    }

    /// Write the set switches onto `o`. `presolve` is not an option field: it
    /// selects [`solve_presolved`], and the caller reads it.
    pub fn apply(&self, o: &mut QpOptions) {
        if let Some(r) = self.reg {
            o.reg = r;
        }
        if let Some(h) = self.hsde {
            o.use_hsde = h;
        }
        if let Some(e) = self.equilibrate {
            o.equilibrate = e;
        }
        if let Some(c) = self.crossover {
            o.crossover = c;
        }
    }
}

/// The PSD screen on the interior-point engine (gh #112): the convex IPM
/// requires a PSD Hessian and returns a silently-wrong `Optimal` at a saddle
/// point without one, so a caller that *knows* `P` is indefinite is refused
/// rather than answered. `Psd` and `Unknown` pass; the verdict itself is the
/// caller's (Python's `check_psd` guard, or
/// [`crate::certify_psd_lower_triangle`]).
pub fn check_ipm_inertia(inertia: HessianInertia) -> Result<(), PipelineError> {
    if inertia == HessianInertia::Indefinite {
        return Err(PipelineError::IndefiniteUnderIpm);
    }
    Ok(())
}

/// What a presolved solve ([`solve_presolved`]) did, for the result's
/// `presolve` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct PresolveReport {
    /// `"reduced"`, `"infeasible"` or `"unbounded"`.
    pub outcome: &'static str,
    /// Present when the problem was reduced.
    pub stats: Option<PresolveStats>,
    /// The screen that proved infeasibility, or the one whose unconfirmed claim
    /// was discarded (gh #523) — `None` when neither happened.
    pub reason: Option<String>,
}

/// Presolve, solve the reduced problem with `solve`, postsolve — the CLI's
/// `run_convex_qp` sequence, minus its `.nl` plumbing. The reduced problem's
/// objective is offset from the original's by `ps.obj_offset()`, which the
/// stopping test is told about through `obj_constant`, and the budget left
/// after presolve is what the solve gets.
///
/// A presolve verdict (`infeasible`, `unbounded`) returns a zero point of the
/// original dimensions with that status and runs no engine.
pub fn solve_presolved<S>(prob: &QpProblem, o: &QpOptions, solve: S) -> (QpSolution, PresolveReport)
where
    S: FnOnce(&QpProblem, &QpOptions) -> QpSolution,
{
    let t0 = std::time::Instant::now();
    let trivial = |status| QpSolution {
        status,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![0.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => {
            let mut ro = QpOptions {
                obj_constant: o.obj_constant + ps.obj_offset(),
                ..*o
            };
            if let Some(limit) = ro.time_limit {
                ro.time_limit = Some(limit.saturating_sub(t0.elapsed()));
            }
            let red = solve(&ps.reduced, &ro);
            let report = PresolveReport {
                outcome: "reduced",
                stats: Some(ps.stats()),
                reason: ps.discarded_infeasibility().map(|t| t.to_string()),
            };
            (ps.postsolve(&red), report)
        }
        PresolveOutcome::Infeasible(trigger) => (
            trivial(QpStatus::PrimalInfeasible),
            PresolveReport {
                outcome: "infeasible",
                stats: None,
                reason: Some(trigger.to_string()),
            },
        ),
        PresolveOutcome::Unbounded => (
            trivial(QpStatus::DualInfeasible),
            PresolveReport {
                outcome: "unbounded",
                stats: None,
                reason: Some("a free column with a nonzero objective coefficient".to_string()),
            },
        ),
    }
}

/// The CLI's post-presolve demotion (gh#984 review): the IPM judged `Optimal`
/// on the *reduced* problem, so re-judge the postsolved point on the problem
/// the caller passed. A demotion only (`Optimal` → `OptimalInaccurate`), and
/// only for the HSDE driver, exactly as `run_convex_qp` applies it.
pub fn demote_unverified_optimal(prob: &QpProblem, o: &QpOptions, sol: QpSolution) -> QpSolution {
    if o.use_hsde
        && sol.status == QpStatus::Optimal
        && sol.kkt_residuals_above_floor(prob).kkt_error() > o.tol
    {
        QpSolution {
            status: QpStatus::OptimalInaccurate,
            ..sol
        }
    } else {
        sol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::solve_qp_ipm;
    use crate::qp::{NEG_INF, POS_INF, Triplet};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// `min ½ (x0² + x1²) − x0 − x1` over `0 ≤ x ≤ 10`, optimum `x = (1, 1)`,
    /// objective `−1`.
    fn box_qp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![10.0, 10.0],
        }
    }

    fn sol_at(prob: &QpProblem, x: Vec<f64>, status: QpStatus) -> QpSolution {
        QpSolution {
            status,
            x,
            y: vec![0.0; prob.m_eq()],
            z: vec![0.0; prob.m_ineq()],
            z_lb: vec![0.0; prob.n],
            z_ub: vec![0.0; prob.n],
            obj: 0.0,
            iters: 3,
            iterates: Vec::new(),
        }
    }

    fn hsde_opts() -> QpOptions {
        QpOptions {
            use_hsde: true,
            ..QpOptions::default()
        }
    }

    // --- demote_unverified_optimal: one fixture per branch -----------------

    #[test]
    fn demotes_an_optimal_whose_point_fails_kkt_under_hsde() {
        let p = box_qp();
        // x = (5, 5): interior, gradient (4, 4) with zero bound duals.
        let s = sol_at(&p, vec![5.0, 5.0], QpStatus::Optimal);
        assert!(s.kkt_residuals_above_floor(&p).kkt_error() > hsde_opts().tol);
        let out = demote_unverified_optimal(&p, &hsde_opts(), s);
        assert_eq!(out.status, QpStatus::OptimalInaccurate);
        // A demotion only: the point is returned untouched.
        assert_eq!(out.x, vec![5.0, 5.0]);
        assert_eq!(out.iters, 3);
    }

    #[test]
    fn keeps_an_optimal_whose_point_passes_kkt() {
        let p = box_qp();
        let s = sol_at(&p, vec![1.0, 1.0], QpStatus::Optimal);
        assert!(s.kkt_residuals_above_floor(&p).kkt_error() <= hsde_opts().tol);
        let out = demote_unverified_optimal(&p, &hsde_opts(), s);
        assert_eq!(out.status, QpStatus::Optimal);
    }

    #[test]
    fn keeps_a_failing_optimal_off_the_hsde_driver() {
        let p = box_qp();
        let o = QpOptions {
            use_hsde: false,
            ..QpOptions::default()
        };
        let out = demote_unverified_optimal(&p, &o, sol_at(&p, vec![5.0, 5.0], QpStatus::Optimal));
        assert_eq!(out.status, QpStatus::Optimal);
    }

    #[test]
    fn leaves_a_non_optimal_status_alone() {
        let p = box_qp();
        let s = sol_at(&p, vec![5.0, 5.0], QpStatus::IterationLimit);
        let out = demote_unverified_optimal(&p, &hsde_opts(), s);
        assert_eq!(out.status, QpStatus::IterationLimit);
    }

    // --- solve_presolved: one fixture per presolve outcome -----------------

    #[test]
    fn presolved_reduced_solves_and_postsolves_to_the_original_space() {
        // x1 is fixed at 2 by its box, which makes the row x0 + x1 ≤ 5 a
        // singleton bound on x0; presolve drops the row and the reduced
        // problem is solved with no inequality rows at all.
        let mut p = box_qp();
        p.lb[1] = 2.0;
        p.ub[1] = 2.0;
        p.g = vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)];
        p.h = vec![5.0];
        let o = QpOptions::default();
        let mut engine_rows = None;
        let (sol, rep) = solve_presolved(&p, &o, |rp, ro| {
            engine_rows = Some(rp.m_ineq());
            solve_qp_ipm(rp, ro, backend)
        });
        assert_eq!(rep.outcome, "reduced");
        assert_eq!(engine_rows, Some(0), "the engine saw the reduced problem");
        let st = rep.stats.expect("a reduced presolve reports stats");
        assert_eq!((st.orig_rows, st.reduced_rows), (1, 0));
        assert_eq!(sol.status, QpStatus::Optimal);
        // Postsolved back to the original dimensions.
        assert_eq!((sol.x.len(), sol.z.len()), (2, 1));
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0 = {}", sol.x[0]);
        assert!((sol.x[1] - 2.0).abs() < 1e-6, "x1 = {}", sol.x[1]);
        // ½·1 − 1 + ½·4 − 2 = −0.5.
        assert!((sol.obj + 0.5).abs() < 1e-6, "obj = {}", sol.obj);
    }

    #[test]
    fn presolved_infeasible_runs_no_engine() {
        // Row x0 + x1 ≤ −1 with x ≥ 0.
        let mut p = box_qp();
        p.g = vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)];
        p.h = vec![-1.0];
        let (sol, rep) = solve_presolved(&p, &QpOptions::default(), |_, _| {
            panic!("the engine must not run on a presolve verdict")
        });
        assert_eq!(rep.outcome, "infeasible");
        assert!(rep.stats.is_none());
        assert!(rep.reason.is_some());
        assert_eq!(sol.status, QpStatus::PrimalInfeasible);
        assert_eq!((sol.x.len(), sol.z.len(), sol.iters), (2, 1, 0));
    }

    #[test]
    fn presolved_unbounded_runs_no_engine() {
        // A free column with a nonzero cost and nothing else.
        let p = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![NEG_INF],
            ub: vec![POS_INF],
        };
        let (sol, rep) = solve_presolved(&p, &QpOptions::default(), |_, _| {
            panic!("the engine must not run on a presolve verdict")
        });
        assert_eq!(rep.outcome, "unbounded");
        assert_eq!(
            rep.reason.as_deref(),
            Some("a free column with a nonzero objective coefficient")
        );
        assert_eq!(sol.status, QpStatus::DualInfeasible);
    }

    // --- the screens: one fixture per refusal and per pass ------------------

    #[test]
    fn engine_switches_refusals_and_passes() {
        let none = EngineSwitches::default();
        assert_eq!(none.check(false, false), Ok(()));
        assert_eq!(none.check(true, true), Ok(()));

        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let s = EngineSwitches {
                reg: Some(bad),
                ..none
            };
            assert!(matches!(
                s.check(false, false),
                Err(PipelineError::InvalidReg(_))
            ));
        }
        let reg_ok = EngineSwitches {
            reg: Some(1e-8),
            ..none
        };
        assert_eq!(reg_ok.check(false, false), Ok(()));
        assert_eq!(
            reg_ok.check(true, false),
            Err(PipelineError::IpmOnlyOption("qp_reg"))
        );
        let hsde = EngineSwitches {
            hsde: Some(false),
            ..none
        };
        assert_eq!(
            hsde.check(true, false),
            Err(PipelineError::IpmOnlyOption("qp_hsde"))
        );
        let cross = EngineSwitches {
            crossover: Some(true),
            ..none
        };
        assert_eq!(
            cross.check(true, false),
            Err(PipelineError::IpmOnlyOption("qp_crossover"))
        );
        assert_eq!(
            cross.check(false, true),
            Err(PipelineError::CrossoverWithWarmStart)
        );
        let pre = EngineSwitches {
            presolve: Some(true),
            ..none
        };
        assert_eq!(pre.check(true, false), Ok(()));
        assert_eq!(
            pre.check(false, true),
            Err(PipelineError::PresolveWithWarmStart)
        );
        // `Some(false)` asks for what a warm start does anyway.
        let off = EngineSwitches {
            presolve: Some(false),
            crossover: Some(false),
            ..none
        };
        assert_eq!(off.check(false, true), Ok(()));
    }

    #[test]
    fn engine_switches_apply_only_what_is_set() {
        let base = QpOptions::default();
        let mut o = base;
        EngineSwitches::default().apply(&mut o);
        assert_eq!(
            (o.reg, o.use_hsde, o.equilibrate, o.crossover),
            (base.reg, base.use_hsde, base.equilibrate, base.crossover)
        );
        EngineSwitches {
            presolve: Some(true),
            reg: Some(0.25),
            hsde: Some(!base.use_hsde),
            equilibrate: Some(!base.equilibrate),
            crossover: Some(!base.crossover),
        }
        .apply(&mut o);
        assert_eq!(o.reg, 0.25);
        assert_eq!(o.use_hsde, !base.use_hsde);
        assert_eq!(o.equilibrate, !base.equilibrate);
        assert_eq!(o.crossover, !base.crossover);
    }

    #[test]
    fn ipm_inertia_screen() {
        assert_eq!(check_ipm_inertia(HessianInertia::Psd), Ok(()));
        assert_eq!(check_ipm_inertia(HessianInertia::Unknown), Ok(()));
        assert_eq!(
            check_ipm_inertia(HessianInertia::Indefinite),
            Err(PipelineError::IndefiniteUnderIpm)
        );
    }
}
