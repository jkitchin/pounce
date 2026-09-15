//! [`ConvexPresolveSession`] is a persistent convex presolve + IPM for QP
//! families solved with warm starts.
//!
//! The IPM takes an original-space [`QpWarmStart`] but presolve solves the
//! reduced problem. This session maps every seed into reduced space before
//! solving, then postsolves back. It retains [`Presolve`] only while the
//! complete numeric [`QpProblem`] is unchanged; changing a cost, RHS, bound,
//! or matrix value rebuilds presolve but still uses the supplied warm point.
//!
//! ```no_run
//! # use pounce_convex::{ConvexPresolveSession, QpOptions, QpProblem};
//! # use pounce_convex::ipm::QpWarmStart;
//! # fn demo(
//! #     mut backend: impl FnMut() -> Box<dyn pounce_linsol::SparseSymLinearSolverInterface> + 'static,
//! #     problems: Vec<QpProblem>,
//! # ) {
//! let mut session = ConvexPresolveSession::new();
//! let mut warm: Option<QpWarmStart> = None;
//! for prob in &problems {
//!     let sol = session.solve(prob, warm.as_ref(), &QpOptions::default(), &mut backend);
//!     warm = Some(QpWarmStart::from_solution(&sol));
//! }
//! # }
//! ```
//!
//! # Cache + validate
//!
//! Each solve fingerprints every number in the problem data
//! ([`convex_presolve_fingerprint`]): an exact match reuses the retained
//! `Presolve` and skips the recompute; any mismatch runs presolve fresh and
//! maps the seed through the new transform. This is intentionally stricter
//! than `TnlpPresolveSession`'s objective-only policy: convex presolve owns the
//! reduced objective and the original numeric problem used for postsolve,
//! rather than delegating those evaluations to a live TNLP.

use pounce_linsol::SparseSymLinearSolverInterface;

use crate::active_set::empty_solution;
use crate::ipm::{QpOptions, QpWarmStart, solve_qp_ipm, solve_qp_ipm_warm};
use crate::presolve::{
    ConvexPresolveFingerprint, ConvexWarmReport, Presolve, PresolveOutcome,
    convex_presolve_fingerprint, presolve,
};
use crate::qp::{QpProblem, QpSolution, QpStatus};

/// Persistent convex solve over [`QpProblem`]s with presolve retained and
/// original-space seeds mapped through it.
///
/// Construct once, [`solve`](Self::solve) many times. Answers come back in
/// the handed problem's coordinates.
///
/// A presolve verdict (`Infeasible`/`Unbounded`) ends the solve with no
/// engine call and clears the retained transform, mirroring
/// [`ActiveSetSession`](crate::ActiveSetSession)'s `NoSolve` leg.
pub struct ConvexPresolveSession {
    presolve: bool,
    retained: Option<Presolve>,
    retained_fp: Option<ConvexPresolveFingerprint>,
    last_reused: bool,
    last_report: ConvexWarmReport,
    solves: usize,
}

impl ConvexPresolveSession {
    /// Open a session with convex presolve on.
    pub fn new() -> Self {
        Self {
            presolve: true,
            retained: None,
            retained_fp: None,
            last_reused: false,
            last_report: ConvexWarmReport::default(),
            solves: 0,
        }
    }

    /// Presolve (and postsolve) around each solve. On by default. Off is for
    /// callers needing the iterate in as-posed coordinates, or families
    /// whose reduction is unstable across members.
    #[must_use]
    pub fn with_presolve(mut self, on: bool) -> Self {
        self.set_presolve(on);
        self
    }

    /// Toggle presolve on a live session (the [`Self::with_presolve`]
    /// form consumes, so it cannot flip a session mid-sweep). Turning it
    /// off drops the retained transform, like [`Self::reset`].
    pub fn set_presolve(&mut self, on: bool) {
        self.presolve = on;
        if !on {
            self.reset();
        }
    }

    /// Drop the retained transform, the next solve runs presolve fresh.
    pub fn reset(&mut self) {
        self.retained = None;
        self.retained_fp = None;
    }

    /// Calls to [`solve`](Self::solve).
    pub fn solves(&self) -> usize {
        self.solves
    }

    /// Whether the last solve reused an exact-numeric-match transform.
    /// Equal sparsity with changed coefficients is a rebuild, not reuse.
    pub fn last_reused_transform(&self) -> bool {
        self.last_reused
    }

    /// What the last solve's warm-point projection did.
    pub fn last_report(&self) -> &ConvexWarmReport {
        &self.last_report
    }

    /// Solve `prob`, seeding from `warm` (original space) when shaped for it.
    ///
    /// A misshaped `warm` solves cold.
    /// `max_iter = 0` stays a frontend semantic (see `ActiveSetSession`).
    pub fn solve<F>(
        &mut self,
        prob: &QpProblem,
        warm: Option<&QpWarmStart>,
        opts: &QpOptions,
        mut make_backend: F,
    ) -> QpSolution
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        self.solves += 1;
        if !self.presolve {
            self.last_reused = false;
            self.last_report = ConvexWarmReport::default();
            return match warm {
                Some(w) => solve_qp_ipm_warm(prob, opts, w, &mut make_backend),
                None => solve_qp_ipm(prob, opts, &mut make_backend),
            };
        }
        let fp = convex_presolve_fingerprint(prob);
        if self.retained_fp.as_ref() == Some(&fp) && self.retained.is_some() {
            self.last_reused = true;
        } else {
            self.last_reused = false;
            match presolve(prob) {
                PresolveOutcome::Reduced(ps) => {
                    self.retained = Some(ps);
                    self.retained_fp = Some(fp);
                }
                PresolveOutcome::Infeasible(_) => {
                    self.retained = None;
                    self.retained_fp = None;
                    self.last_report = ConvexWarmReport::default();
                    return empty_solution(
                        prob.n,
                        prob.m_eq(),
                        prob.m_ineq(),
                        QpStatus::PrimalInfeasible,
                    );
                }
                PresolveOutcome::Unbounded => {
                    self.retained = None;
                    self.retained_fp = None;
                    self.last_report = ConvexWarmReport::default();
                    return empty_solution(
                        prob.n,
                        prob.m_eq(),
                        prob.m_ineq(),
                        QpStatus::DualInfeasible,
                    );
                }
            }
        }
        let ps = match self.retained.as_ref() {
            Some(ps) => ps,
            // Unreachable (every arm above populates it), solve cold rather
            // than panic.
            None => {
                return match warm {
                    Some(w) => solve_qp_ipm_warm(prob, opts, w, &mut make_backend),
                    None => solve_qp_ipm(prob, opts, &mut make_backend),
                };
            }
        };
        // Reduced objective differs by `obj_offset()`.
        let red_opts = QpOptions {
            obj_constant: opts.obj_constant + ps.obj_offset(),
            ..*opts
        };
        let red = match warm {
            Some(w) => match ps.project_warm(w) {
                Some((projected, report)) => {
                    self.last_report = report;
                    solve_qp_ipm_warm(&ps.reduced, &red_opts, &projected, &mut make_backend)
                }
                None => {
                    self.last_report = ConvexWarmReport::default();
                    solve_qp_ipm(&ps.reduced, &red_opts, &mut make_backend)
                }
            },
            None => {
                self.last_report = ConvexWarmReport::default();
                solve_qp_ipm(&ps.reduced, &red_opts, &mut make_backend)
            }
        };
        ps.postsolve(&red)
    }
}

impl Default for ConvexPresolveSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// min x0^2 + x1^2  s.t.  x0 == 0.5 (singleton: fixes x0), x0 + x1 == 1.
    /// Presolve pins x0, leaving x1 == 0.5. Optimum (0.5, 0.5).
    fn fixed_var_qp() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 1.0),
            ],
            b: vec![0.5, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        }
    }

    #[test]
    fn fingerprint_stable_then_moves_with_data() {
        let p = fixed_var_qp();
        let a = convex_presolve_fingerprint(&p);
        let b = convex_presolve_fingerprint(&p);
        assert_eq!(a, b);
        let mut q = fixed_var_qp();
        q.c[0] = 1.0;
        assert_ne!(a, convex_presolve_fingerprint(&q));
    }

    #[test]
    fn project_warm_gathers_survivors() {
        let prob = fixed_var_qp();
        let ps = match presolve(&prob) {
            PresolveOutcome::Reduced(ps) => ps,
            PresolveOutcome::Infeasible(_) => panic!("expected reduction, got infeasible"),
            PresolveOutcome::Unbounded => panic!("expected reduction, got unbounded"),
        };
        assert!(
            ps.reduced.n < prob.n,
            "test needs a column removal, stats = {:?}",
            ps.stats()
        );
        let warm = QpWarmStart {
            x: vec![0.5, 0.5],
            y: vec![1.0, 2.0],
            z: vec![],
            z_lb: vec![0.0, 0.0],
            z_ub: vec![0.0, 0.0],
        };
        let (projected, report) = ps.project_warm(&warm).expect("shaped");
        assert_eq!(projected.x.len(), ps.reduced.n);
        assert_eq!(projected.y.len(), ps.reduced.m_eq());
        assert!(report.layers >= 1, "report = {report:?}");

        let bad = QpWarmStart {
            x: vec![0.5],
            ..warm.clone()
        };
        assert!(ps.project_warm(&bad).is_none());
    }

    #[test]
    fn cold_then_warm_through_session() {
        let prob = fixed_var_qp();
        let opts = QpOptions::default();
        let mut session = ConvexPresolveSession::new();

        let cold = session.solve(&prob, None, &opts, backend);
        assert_eq!(cold.status, QpStatus::Optimal);
        assert!((cold.x[0] - 0.5).abs() < 1e-6, "x = {:?}", cold.x);
        assert!((cold.x[1] - 0.5).abs() < 1e-6, "x = {:?}", cold.x);
        assert!(!session.last_reused_transform(), "first solve builds");

        // Identical data reuses the retained transform.
        let warm = QpWarmStart::from_solution(&cold);
        let second = session.solve(&prob, Some(&warm), &opts, backend);
        assert_eq!(second.status, QpStatus::Optimal);
        assert!(session.last_reused_transform(), "identical data reuses");
        assert!(
            session.last_report().layers >= 1,
            "report = {:?}",
            session.last_report()
        );
        assert!((second.x[0] - 0.5).abs() < 1e-6);
        assert!((second.x[1] - 0.5).abs() < 1e-6);
        assert_eq!(session.solves(), 2);

        // Convex presolve owns a numeric QP snapshot, so even a pure cost
        // change rebuilds the transform while still serving the warm point.
        let mut prob2 = fixed_var_qp();
        prob2.c = vec![-1.0, 0.0];
        let third = session.solve(&prob2, Some(&warm), &opts, backend);
        assert_eq!(third.status, QpStatus::Optimal);
        assert!(!session.last_reused_transform(), "moved data rebuilds");
        // min x0^2 + x1^2 - x0  s.t.  x0 = 0.5, x0 + x1 = 1 → (0.5, 0.5).
        assert!((third.x[0] - 0.5).abs() < 1e-6, "x = {:?}", third.x);
        assert!((third.x[1] - 0.5).abs() < 1e-6, "x = {:?}", third.x);
    }

    #[test]
    fn presolve_off_still_solves_warm() {
        let prob = fixed_var_qp();
        let opts = QpOptions::default();
        let mut session = ConvexPresolveSession::new().with_presolve(false);
        let cold = session.solve(&prob, None, &opts, backend);
        assert_eq!(cold.status, QpStatus::Optimal);
        let warm = QpWarmStart::from_solution(&cold);
        let second = session.solve(&prob, Some(&warm), &opts, backend);
        assert_eq!(second.status, QpStatus::Optimal);
        assert!(!session.last_reused_transform());
    }
}
