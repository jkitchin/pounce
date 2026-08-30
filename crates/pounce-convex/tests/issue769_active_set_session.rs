//! gh #769 — `ActiveSetSession`: the convex → `pounce-qp` translation, the
//! presolve/postsolve wrapper, and parametric reuse, owned in one place.
//!
//! Three properties are what make the session worth having, and each is a way
//! it could silently go wrong:
//!
//! 1. **A session that cannot reuse anything is the free function.** The whole
//!    retry ladder — screen, unscaled attempt, Ruiz retry, simplex-seeded
//!    retry, certificate re-derivation — is reached by calling into the driver
//!    rather than restated, so a cold session solve must be bit-identical to
//!    [`solve_qp_active_set`]. A session that had quietly become a second
//!    implementation would still look fine on any single problem.
//! 2. **Reuse changes the cost, never the verdict.** A warm answer reaches the
//!    caller only through the same verification a cold one does, so a swept
//!    family must produce the *same answers* it would have produced cold —
//!    including when a member of the sweep stops being solvable.
//! 3. **Presolve is inside.** Every frontend was open-coding the
//!    presolve → solve → postsolve wrapper (or skipping it), so the reported
//!    iterate must come back in the coordinates of the problem as posed, with
//!    the objective offset the reduction moved carried back too.

use pounce_convex::{
    ActiveSetOverrides, ActiveSetQp, ActiveSetSession, BoxScreen, HessianInertia, PresolveNote,
    QpOptions, QpProblem, QpSolution, QpStatus, Reuse, Triplet, back_translate,
    back_translate_verified, engine_options, screen_variable_box, solve_qp_active_set,
    verify_status,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// The free function, under the session's own defaults.
fn cold(prob: &QpProblem) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
}

fn session() -> ActiveSetSession {
    ActiveSetSession::new(backend)
}

/// `min (x₀−a)² + (x₁−b)²  s.t.  x₀ + x₁ ≤ 1,  0 ≤ x ≤ 10`, in `½xᵀPx + cᵀx`
/// form with the constant dropped.
///
/// A one-parameter family in `(a, b)`: `P`, `A`, `G` and the box are fixed and
/// only `c` moves, which is the shape [`QpSolver::solve_parametric`] traces.
/// The row binds for `a + b > 1` and is slack below it, so a sweep crosses an
/// active-set change rather than staying on one face.
fn target_qp(a: f64, b: f64) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0 * a, -2.0 * b],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
        h: vec![1.0],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    }
}

fn assert_same_solution(warm: &QpSolution, cold: &QpSolution, what: &str) {
    assert_eq!(warm.status, cold.status, "{what}: status");
    assert!(
        (warm.obj - cold.obj).abs() <= 1e-7 * (1.0 + cold.obj.abs()),
        "{what}: obj {} vs {}",
        warm.obj,
        cold.obj
    );
    for (i, (&w, &c)) in warm.x.iter().zip(cold.x.iter()).enumerate() {
        assert!((w - c).abs() <= 1e-6, "{what}: x{i} {w} vs {c}");
    }
    for (i, (&w, &c)) in warm.z.iter().zip(cold.z.iter()).enumerate() {
        assert!((w - c).abs() <= 1e-6, "{what}: z{i} {w} vs {c}");
    }
}

/// Property 1: with nothing to reuse and presolve off, the session *is* the
/// free function — every field, not just the status and objective.
#[test]
fn session_cold_matches_free_function() {
    let mut s = session().with_presolve(false);
    for (a, b) in [(3.0, 2.0), (0.2, 0.3), (5.0, -1.0)] {
        let prob = target_qp(a, b);
        let got = s.solve_cold(&prob);
        let want = cold(&prob);
        assert_eq!(got.status, want.status, "status at ({a}, {b})");
        assert_eq!(got.x, want.x, "x at ({a}, {b})");
        assert_eq!(got.y, want.y, "y at ({a}, {b})");
        assert_eq!(got.z, want.z, "z at ({a}, {b})");
        assert_eq!(got.z_lb, want.z_lb, "z_lb at ({a}, {b})");
        assert_eq!(got.z_ub, want.z_ub, "z_ub at ({a}, {b})");
        assert_eq!(got.obj, want.obj, "obj at ({a}, {b})");
        assert_eq!(got.iters, want.iters, "iters at ({a}, {b})");
        assert_eq!(s.last_reuse(), Reuse::Cold);
    }
    assert_eq!(s.stats().solves, 3);
    assert_eq!(s.stats().cold_solves, 3);
    assert_eq!(s.stats().parametric_attempts, 0);
}

/// Property 2: a swept family reuses, and every warm answer matches the cold
/// one it replaced.
#[test]
fn parametric_reuse_engages_and_agrees_with_cold() {
    let mut s = session().with_presolve(false);
    // Deliberately crosses the point where the row starts to bind (`a + b = 1`)
    // so the sweep is not one long stay on a single active set.
    let sweep = [
        (0.2, 0.3),
        (0.4, 0.4),
        (0.9, 0.4),
        (3.0, 2.0),
        (5.0, 5.0),
        (0.1, 0.1),
    ];
    for (a, b) in sweep {
        let prob = target_qp(a, b);
        let got = s.solve(&prob);
        assert_same_solution(&got, &cold(&prob), &format!("({a}, {b})"));
        assert_eq!(got.status, QpStatus::Optimal, "({a}, {b})");
    }
    let st = s.stats();
    assert_eq!(st.solves, 6);
    assert!(
        st.parametric_attempts >= 5,
        "every member after the first is eligible: {st:?}"
    );
    // The route, not just the count. This family is the one the homotopy is
    // for — same `P`, same rows, only `c` moving — so every accepted attempt
    // must report the traced path. Asserting `attempts_accepted()` here would
    // pass just as well if the engine had declined all five and fallen back to
    // the working-set hint, which is the hole @GermanHeim found in the first
    // version of this test (gh #769).
    assert_eq!(
        st.homotopy_accepted, 5,
        "the family is exactly the one the homotopy traces: {st:?}"
    );
    assert_eq!(st.working_set_accepted, 0, "{st:?}");
    assert_eq!(st.engine_cold_accepted, 0, "{st:?}");
    assert_eq!(s.last_reuse(), Reuse::Homotopy);
}

/// The first solve of a session has nothing to trace from, and says so.
#[test]
fn the_first_solve_is_cold() {
    let mut s = session();
    let sol = s.solve(&target_qp(3.0, 2.0));
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(s.last_reuse(), Reuse::Cold);
    assert_eq!(s.stats().parametric_attempts, 0);
}

/// [`ActiveSetSession::reset`] drops the pair, and the next solve is cold
/// again — the control a caller reaches for when the next problem is unrelated.
#[test]
fn reset_forces_the_next_solve_cold() {
    let mut s = session().with_presolve(false);
    s.solve(&target_qp(3.0, 2.0));
    s.reset();
    let sol = s.solve(&target_qp(2.0, 2.0));
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(s.last_reuse(), Reuse::Cold);
    assert_eq!(s.stats().parametric_attempts, 0);
}

/// A shape change is the one eligibility question the session answers itself,
/// because `solve_parametric` would answer it with a cold solve that has
/// neither the Ruiz retry nor the simplex seed behind it.
#[test]
fn a_shape_change_falls_back_to_the_cold_ladder() {
    let mut s = session().with_presolve(false);
    s.solve(&target_qp(3.0, 2.0));

    // Same family, one more variable: `min Σ(xᵢ−1)² s.t. Σxᵢ ≤ 1`.
    let bigger = QpProblem {
        n: 3,
        p_lower: (0..3).map(|i| Triplet::new(i, i, 2.0)).collect(),
        c: vec![-2.0; 3],
        a: vec![],
        b: vec![],
        g: (0..3).map(|j| Triplet::new(0, j, 1.0)).collect(),
        h: vec![1.0],
        lb: vec![0.0; 3],
        ub: vec![10.0; 3],
    };
    let got = s.solve(&bigger);
    assert_eq!(s.last_reuse(), Reuse::Cold, "shape changed");
    assert_eq!(s.stats().parametric_attempts, 0);
    assert_same_solution(&got, &cold(&bigger), "n = 3");
    // `x* = (1/3, 1/3, 1/3)` by symmetry on the binding row.
    for (i, &xi) in got.x.iter().enumerate() {
        assert!((xi - 1.0 / 3.0).abs() < 1e-6, "x{i} = {xi}");
    }
}

/// Property 2, the hard half: a member of the family that is *infeasible* must
/// still come back certified `PrimalInfeasible`, not carried along by the
/// previous member's answer.
///
/// The shape is deliberately unchanged from the member before it, so this goes
/// through the **warm** path rather than falling back on a shape mismatch —
/// and the certificate is re-derived against the problem as posed, so reuse
/// cannot manufacture one.
#[test]
fn an_infeasible_member_is_still_certified() {
    let mut s = session().with_presolve(false);
    s.solve(&target_qp(3.0, 2.0));

    // Same `P`, `G` and box; only `h` moves — `x₀ + x₁ ≤ −5` cannot be met
    // inside `0 ≤ x`.
    let mut infeasible = target_qp(3.0, 2.0);
    infeasible.h = vec![-5.0];

    let got = s.solve(&infeasible);
    assert_eq!(s.last_reuse(), Reuse::Homotopy, "reached through reuse");
    assert_eq!(got.status, QpStatus::PrimalInfeasible);
    assert_same_solution(&got, &cold(&infeasible), "infeasible member");
}

/// A warm verdict that does not stand up is not reported, and the cold ladder
/// owns the answer. `tol = 1e-300` is unreachable in double precision, so
/// nothing can be certified and every attempt is rejected — the wiring under
/// test is that a rejection *falls through* rather than reporting the
/// unverified point.
#[test]
fn a_rejected_warm_verdict_falls_through_to_the_cold_ladder() {
    let mut s = session().with_presolve(false);
    s.solve(&target_qp(3.0, 2.0));
    s.set_options(QpOptions {
        tol: 1e-300,
        ..QpOptions::default()
    });

    let prob = target_qp(2.5, 2.0);
    let got = s.solve(&prob);
    assert_eq!(s.last_reuse(), Reuse::ParametricRejected);
    assert_eq!(s.stats().parametric_attempts, 1);
    assert_eq!(s.stats().attempts_accepted(), 0);
    // Two cold solves: the first member, and the fallback for this one.
    assert_eq!(s.stats().cold_solves, 2);
    // The reported answer is the cold ladder's, honest failure and all.
    let want = {
        let mut mk = backend;
        solve_qp_active_set(
            &prob,
            &QpOptions {
                tol: 1e-300,
                ..QpOptions::default()
            },
            &ActiveSetOverrides::default(),
            &mut mk,
        )
    };
    assert_eq!(got.status, want.status);
    assert_ne!(got.status, QpStatus::Optimal);
}

/// The variable-box screen runs on the warm path too. An empty box panicked the
/// engine (gh #295); a session that skipped the screen because it "already
/// solved this family" would reach the same panic.
#[test]
fn the_box_screen_runs_before_a_warm_solve() {
    let mut s = session().with_presolve(false);
    s.solve(&target_qp(3.0, 2.0));

    let mut crossed = target_qp(3.0, 2.0);
    crossed.lb = vec![1.0, 0.0];
    crossed.ub = vec![0.0, 10.0];
    let got = s.solve(&crossed);
    assert_eq!(got.status, QpStatus::PrimalInfeasible);
    // The screen declined the warm attempt rather than handing the engine an
    // empty box, and the certified status came from the one place that
    // produces it.
    assert_eq!(s.last_reuse(), Reuse::Cold);
    assert_eq!(s.stats().parametric_attempts, 0);
}

/// Property 3: presolve is inside the session, and the answer comes back in
/// the coordinates of the problem as posed.
///
/// `min (x₀−3)² + (x₁−2)²  s.t.  x₀ = 1`, box `[0, 10]²`. The singleton
/// equality row fixes `x₀`, so presolve removes both the row and the variable
/// and hands the engine a one-variable problem plus an objective offset of
/// `(1−3)² = 4` — the term a postsolve that dropped the offset would lose.
#[test]
fn presolve_and_postsolve_are_owned_by_the_session() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        // (x₀−3)² + (x₁−2)² = ½xᵀPx + cᵀx + 13, constant dropped.
        c: vec![-6.0, -4.0],
        a: vec![Triplet::new(0, 0, 1.0)],
        b: vec![1.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };
    let mut s = session();
    let sol = s.solve(&prob);

    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(sol.x.len(), 2, "reported in the original space");
    assert!((sol.x[0] - 1.0).abs() < 1e-8, "x0 = {}", sol.x[0]);
    assert!((sol.x[1] - 2.0).abs() < 1e-8, "x1 = {}", sol.x[1]);
    // ½xᵀPx + cᵀx at (1, 2) = (1 + 4) + (−6 − 8) = −9.
    assert!((sol.obj + 9.0).abs() < 1e-7, "obj = {}", sol.obj);

    match s.last_presolve() {
        PresolveNote::Reduced { stats, .. } => {
            assert!(
                stats.reduced_vars < stats.orig_vars || stats.reduced_rows < stats.orig_rows,
                "presolve should have removed the fixed variable: {stats:?}"
            );
        }
        other => panic!("expected a reduction, got {other:?}"),
    }
}

/// A presolve verdict is reported with no solve behind it, and leaves nothing
/// to warm-start the next problem from.
#[test]
fn presolve_infeasibility_is_reported_without_a_solve() {
    // `x₀ = 1` and `x₀ = 2`: two singleton equality rows on the same variable.
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 0, 1.0)],
        b: vec![1.0, 2.0],
        g: vec![],
        h: vec![],
        lb: vec![0.0],
        ub: vec![10.0],
    };
    let mut s = session();
    let sol = s.solve(&prob);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    assert_eq!(s.last_reuse(), Reuse::NoSolve);
    assert_eq!(s.stats().cold_solves, 0, "no engine solve ran");
    assert!(
        matches!(s.last_presolve(), PresolveNote::Infeasible { .. }),
        "got {:?}",
        s.last_presolve()
    );

    // And the next solve starts cold rather than tracing from a pair that
    // describes some earlier problem.
    let sol = s.solve(&target_qp(3.0, 2.0));
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(s.last_reuse(), Reuse::Cold);
}

/// Options set on the session reach the engine. `max_iter = 1` is not enough
/// for this QP's active-set changes, and the honest report is the iteration
/// limit — not a warm-started `Optimal` from the previous member.
#[test]
fn options_reach_the_engine_and_a_budget_is_not_warm_started_around() {
    let mut s = session().with_presolve(false).with_options(QpOptions {
        max_iter: 1,
        ..QpOptions::default()
    });
    let sol = s.solve(&target_qp(3.0, 2.0));
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "a one-iteration budget cannot certify this QP"
    );
}

/// The engine declined the path; the session must say the engine declined the
/// path.
///
/// `solve_parametric` guards on an unchanged `H` — it interpolates `g` and the
/// row bounds along the path, not the Hessian — so a member whose curvature
/// moved is answered from the previous *working set* instead. That is still a
/// warm solve and still the right answer, but it is not the homotopy, and the
/// first version of this session reported it as one: every conclusive return
/// counted as `parametric_accepted`, so a family that never traced a single
/// path reported perfect reuse. Reproduced by @GermanHeim in review of
/// gh #769 with exactly this change (`P`'s diagonal 2 → 4).
#[test]
fn a_declined_homotopy_is_reported_as_a_working_set_reuse() {
    let mut s = session().with_presolve(false);
    s.solve(&target_qp(3.0, 2.0));
    assert_eq!(s.last_reuse(), Reuse::Cold, "first solve has no base");

    // Same shape, same rows, same box — only the curvature moves.
    let mut stiffer = target_qp(3.0, 2.0);
    stiffer.p_lower = vec![Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 4.0)];

    let got = s.solve(&stiffer);
    assert_eq!(
        s.last_reuse(),
        Reuse::WorkingSet,
        "a changed Hessian declines the path — reporting `Homotopy` here is \
         the defect this test exists for"
    );
    assert!(s.last_reuse().is_warm(), "the working set did carry over");

    let st = s.stats();
    assert_eq!(st.parametric_attempts, 1, "{st:?}");
    assert_eq!(st.homotopy_accepted, 0, "the path was never traced: {st:?}");
    assert_eq!(st.working_set_accepted, 1, "{st:?}");
    assert_eq!(st.attempts_accepted(), 1, "{st:?}");
    assert_eq!(st.warm_accepted(), 1, "{st:?}");

    // Declined or not, the answer is the cold one.
    assert_same_solution(&got, &cold(&stiffer), "stiffer member");
}

/// The session never hands the engine a base it will discard, so
/// [`Reuse::EngineCold`] should not be reachable through it.
///
/// `solve_parametric` solves cold when the previous status is not `Optimal` or
/// its working set does not fit the new problem — and the session's `remember`
/// keeps a pair only when the engine *and* the driver called it `Optimal`,
/// while its own shape guard covers the dimensions. The variant exists anyway
/// because that decision belongs to the engine's guards, which can grow: what
/// must never happen is the session inventing a route the engine did not
/// report. This pins the current state so a future engine change that starts
/// cold-solving inside `solve_parametric` shows up here rather than as a
/// silently inflated reuse count.
#[test]
fn the_session_never_reports_a_route_the_engine_did_not_take() {
    let mut s = session().with_presolve(false);
    for (a, b) in [(3.0, 2.0), (2.0, 1.0), (0.4, 0.4), (5.0, 5.0)] {
        s.solve(&target_qp(a, b));
        assert_ne!(
            s.last_reuse(),
            Reuse::EngineCold,
            "({a}, {b}): the session only warm-starts from an Optimal base"
        );
    }
    let st = s.stats();
    assert_eq!(st.engine_cold_accepted, 0, "{st:?}");
    assert_eq!(
        st.attempts_accepted(),
        st.warm_accepted(),
        "every accepted attempt reused something: {st:?}"
    );
}

/// The return leg of the translation is public, and composing it correctly is
/// one call.
///
/// This is the second review finding on gh #769: `ActiveSetQp` shipped with the
/// forward translation public and the read-back crate-private, so an external
/// driver (oximo, the case in the issue) could build and solve the native
/// problem and then had to restate the dual sign transform, the objective
/// reconstruction and the verification gate — the three parts that fail
/// silently. This test is written the way such a caller writes it: nothing but
/// the crate's public API, no session and no free function.
#[test]
fn an_external_caller_can_translate_solve_and_read_back() {
    use pounce_qp::{ParametricActiveSetSolver, QpSolver};

    let prob = target_qp(3.0, 2.0);
    let opts = QpOptions::default();

    // Step 1 of the recipe, and the one that is not optional: the box screen.
    let prob = match screen_variable_box(&prob) {
        BoxScreen::Feasible => prob,
        BoxScreen::Snapped(repaired) => repaired,
        BoxScreen::Empty => unreachable!("this fixture's box is fine"),
    };

    let native = ActiveSetQp::from_convex(&prob);
    let qopts = engine_options(
        &opts,
        &ActiveSetOverrides::default(),
        native.n(),
        native.m(),
        // The convex claim, which `from_convex` also attaches by default. The
        // two must agree: `engine_options` reads it for the Schur-update
        // choice and `problem()` hands it to the engine (gh #786).
        HessianInertia::Psd,
    );
    let qsol = ParametricActiveSetSolver::new(backend())
        .solve(&native.problem(), None, &qopts)
        .expect("native solve");

    let got = back_translate_verified(&prob, &qsol, &opts);
    assert_eq!(got.status, QpStatus::Optimal);
    assert_same_solution(&got, &cold(&prob), "external caller");

    // The pieces are exported too, and the composition is exactly them.
    let mut by_hand = back_translate(&prob, &qsol);
    by_hand.status = verify_status(
        qsol.status,
        qsol.unbounded_ray.as_deref(),
        qsol.stats.second_order,
        &by_hand,
        &prob,
        &opts,
    );
    assert_eq!(by_hand.status, got.status, "composition matches its pieces");
    assert_eq!(by_hand.x, got.x);
    assert_eq!(by_hand.z, got.z);
}

/// The screen an external caller must run is reachable, and it is the step
/// that decides a class the engine cannot.
///
/// `solve_qp_active_set` and `ActiveSetSession` screen the variable box before
/// translating, because two classes of empty box do not survive the engine:
/// a reversed box (`lb > ub`) is rejected by `pounce_qp`'s `validate` as a hard
/// `Err`, and a *present* `+∞` lower bound is dropped as if absent and comes
/// back `Optimal` at a point violating it (gh #295, gh #491). Exporting the
/// translation without exporting the screen would have left an external caller
/// to rediscover both — the same half-a-surface problem @GermanHeim raised
/// about the read-back, one step earlier in the recipe.
#[test]
fn the_box_screen_is_reachable_and_decides_what_the_engine_cannot() {
    // Reversed beyond the hairline tolerance: empty by inspection.
    let mut reversed = target_qp(3.0, 2.0);
    reversed.lb = vec![5.0, 0.0];
    reversed.ub = vec![1.0, 10.0];
    assert!(
        matches!(screen_variable_box(&reversed), BoxScreen::Empty),
        "a reversed box is empty by inspection, and needs no certificate"
    );
    // And that is the verdict the driver reports for it, rather than the
    // `InvertedBounds` error the engine would raise on the raw translation.
    assert_eq!(cold(&reversed).status, QpStatus::PrimalInfeasible);

    // A hairline crossing is a tolerance artifact, not an empty set: the screen
    // repairs it and the solve proceeds on the repaired copy.
    let mut hairline = target_qp(3.0, 2.0);
    hairline.lb = vec![0.5 + 1e-12, 0.0];
    hairline.ub = vec![0.5, 10.0];
    match screen_variable_box(&hairline) {
        BoxScreen::Snapped(repaired) => {
            assert!(
                repaired.lb_of(0) <= repaired.ub_of(0),
                "repaired to a point"
            );
        }
        other => panic!("hairline crossing must snap, got {other:?}"),
    }
    assert_eq!(cold(&hairline).status, QpStatus::Optimal);
}
