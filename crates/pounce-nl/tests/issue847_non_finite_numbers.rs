//! The `.nl` reader must refuse non-finite numbers, not silently drop them
//! (gh #847).
//!
//! # The mechanism
//!
//! `str::parse::<f64>()` accepts `inf`, `-inf` and `nan`, and it also
//! *returns* `inf` for a literal that overflows the type — `1e400` is a
//! plausible thing for a model generator to emit. The reader had no finiteness
//! screen, and downstream `pounce_common::types::lower_bound_present` /
//! `upper_bound_present` are `is_finite() && ...`, so a non-finite bound is
//! **indistinguishable from a bound that was never declared** and is dropped.
//!
//! The reported consequence, on a model whose feasible region a lower bound
//! makes empty:
//!
//! ```text
//!   lower bound = 100     -> infeasible (correct)
//!   lower bound = 1e18    -> infeasible (correct)
//!   lower bound = 1e300   -> infeasible (correct)
//!   lower bound = 1e400   -> EXIT: Optimal Solution Found., rc 0
//!   lower bound = inf     -> EXIT: Optimal Solution Found., rc 0
//! ```
//!
//! The transition sits exactly at the finiteness edge, not at any modelling
//! scale. A `nan` is worse still: it reaches the answer, and the solve reports
//! `Objective: nan` under `Solve_Succeeded` with exit code 0.
//!
//! # Why this is a defect and not a tolerance opinion
//!
//! POUNCE already refuses this exact input on another surface — `solve_qp`
//! rejects a non-finite bound with a bespoke `ValueError` — so the project's
//! own position on the value is "reject", and the `.nl` path simply did not
//! implement it. Ipopt, which POUNCE ports, refuses it too ("Invalid number").
//!
//! # The one non-finite value that is *not* an error
//!
//! `.nl` states "no bound" with a bound **kind** (1 = upper only, 2 = lower
//! only, 3 = free), so a non-finite number in a bound slot is a corrupt value
//! rather than a notation — with one exception per side. `-inf` in a *lower*
//! slot and `+inf` in an *upper* one say exactly what the `±1e19` sentinel
//! says. Those are normalized; the asymmetry is load-bearing, because `+inf`
//! as a *lower* bound is the reported case and is an empty box, not an absent
//! bound.

use pounce_nl::nl_reader::parse_nl_text;

/// `min x^2` over `lo <= x <= 3`, with the lower bound spliced in as text so
/// the literal reaches the parser exactly as a file would carry it.
fn model_with_lower_bound(lo: &str) -> String {
    format!(
        "g3 1 1 0\t# problem b847\n\
         \x201 0 1 0 0 \t# vars, constraints, objectives, ranges, eqns\n\
         \x200 1\t# nonlinear constraints, objectives\n\
         \x200 0\t# network constraints: nonlinear, linear\n\
         \x200 1 0 \t# nonlinear vars in constraints, objectives, both\n\
         \x200 0 0 1\t# linear network variables; functions; arith, flags\n\
         \x200 0 0 0 0 \t# discrete variables: binary, integer, nonlinear\n\
         \x200 1 \t# nonzeros in Jacobian, obj. gradient\n\
         \x201 1\t# max name lengths: constraints, variables\n\
         \x200 0 0 0 0\t# common exprs: b,c,o,c1,o1\n\
         O0 0\t#o\n\
         o5\t#^\n\
         v0\t#x\n\
         n2\n\
         x1\t# initial guess\n\
         0 4.0\t#x\n\
         r\t#0 ranges\n\
         b\t#1 bounds (on variables)\n\
         0 {lo} 3\t#x\n\
         k0\n\
         G0 1\t#o\n\
         0 0\n"
    )
}

/// `min x^2 + <literal>` over `-10 <= x <= 10`.
fn model_with_objective_literal(lit: &str) -> String {
    format!(
        "g3 1 1 0\t# problem o847\n\
         \x201 0 1 0 0 \n\
         \x200 1\n\
         \x200 0\n\
         \x200 1 0 \n\
         \x200 0 0 1\n\
         \x200 0 0 0 0 \n\
         \x200 1 \n\
         \x201 1\n\
         \x200 0 0 0 0\n\
         O0 0\t#o\n\
         o0\t#+\n\
         o5\t#^\n\
         v0\t#x\n\
         n2\n\
         n{lit}\n\
         x1\n\
         0 4.0\t#x\n\
         r\n\
         b\n\
         0 -10 10\t#x\n\
         k0\n\
         G0 1\t#o\n\
         0 0\n"
    )
}

/// The premise: a *finite* bound of the same shape parses, and the box it
/// describes is the crossed one the reported model relies on. Without this the
/// rejections below could be rejecting a malformed fixture.
#[test]
fn the_finite_model_parses_and_carries_the_crossed_box() {
    let prob = parse_nl_text(&model_with_lower_bound("1e300")).expect("finite bound parses");
    assert_eq!(prob.x_l[0], 1e300, "the lower bound survives verbatim");
    assert_eq!(prob.x_u[0], 3.0);
    assert!(
        prob.x_l[0] > prob.x_u[0],
        "premise: this box is empty, which is what makes the model infeasible"
    );
}

/// The reported threshold sweep. Everything finite parses and keeps its
/// number; everything non-finite is refused. The pair `1e300` / `1e400` is the
/// heart of it — one decade apart in the file, and the second one overflows.
#[test]
fn the_finiteness_edge_is_where_the_reader_stops_accepting() {
    for lo in ["100", "1e18", "1e19", "1e300"] {
        let prob = parse_nl_text(&model_with_lower_bound(lo))
            .unwrap_or_else(|e| panic!("finite lower bound {lo} must parse, got {e}"));
        let want: f64 = lo.parse().unwrap();
        assert_eq!(prob.x_l[0], want, "lower bound {lo} must survive verbatim");
    }
    for lo in ["1e400", "inf", "Inf", "infinity", "nan", "NaN", "-nan"] {
        let err = parse_nl_text(&model_with_lower_bound(lo)).expect_err(&format!(
            "a lower bound of {lo} must be refused (the issue reports \
             EXIT: Optimal Solution Found. with exit code 0)"
        ));
        assert!(
            err.contains("invalid number"),
            "lower bound {lo}: the error should say what is wrong, got {err}"
        );
    }
}

/// The asymmetry, which is the part a reader is most likely to get backwards.
/// `-inf` as a *lower* bound is the `±1e19` sentinel said differently and is
/// normalized; `+inf` as a lower bound is an empty box and is refused. The
/// mirror holds on the upper side.
#[test]
fn an_infinity_is_a_sentinel_on_its_own_side_and_an_error_on_the_other() {
    let prob = parse_nl_text(&model_with_lower_bound("-inf"))
        .expect("-inf is a lower bound saying `unbounded below`");
    assert_eq!(
        prob.x_l[0], -1e19,
        "it normalizes to the sentinel the rest of the reader emits"
    );
    let err = parse_nl_text(&model_with_lower_bound("inf"))
        .expect_err("+inf as a *lower* bound is an empty box, not an absent bound");
    assert!(err.contains("invalid number"), "got {err}");
}

/// A non-finite numeric literal in the objective body. This one is worse than
/// a dropped bound because it does not merely change the model, it propagates
/// into the reported answer: the issue records `Objective: nan nan`,
/// `EXIT: Optimal Solution Found.`, `Status: Solve_Succeeded`, `$? = 0`.
#[test]
fn a_non_finite_objective_literal_is_refused() {
    let prob = parse_nl_text(&model_with_objective_literal("7.5")).expect("finite literal parses");
    assert_eq!(prob.n, 1);
    for lit in ["nan", "inf", "-inf", "1e400"] {
        let err = parse_nl_text(&model_with_objective_literal(lit))
            .expect_err(&format!("objective literal {lit} must be refused"));
        assert!(
            err.contains("invalid number"),
            "objective literal {lit}: got {err}"
        );
    }
}

/// The reader has a quadratic-recognition fast path that reads `n` tokens
/// separately from the general expression parser, so a screen on one is not a
/// screen on the other. Both legs are asked here: `use_quadratic` on is the
/// default the CLI takes, off is what `POUNCE_DBG_NO_QUAD` selects.
#[test]
fn both_expression_paths_refuse_it_not_just_the_general_one() {
    for use_quadratic in [true, false] {
        let err = pounce_nl::nl_reader::parse_nl_text_with_quadratic(
            &model_with_objective_literal("nan"),
            use_quadratic,
        )
        .expect_err(&format!("use_quadratic={use_quadratic} must refuse nan"));
        assert!(
            err.contains("invalid number"),
            "use_quadratic={use_quadratic}: got {err}"
        );
    }
}

/// The issue's closing note: the two presence predicates are load-bearing
/// beyond the file reader, so *any* caller that builds a model programmatically
/// and passes a non-finite bound hits the same silent drop.
#[test]
fn the_programmatic_builder_refuses_it_too() {
    use pounce_nl::nl_reader::{Expr, NlProblem, NlProblemParts};

    let parts = |x_l: Vec<f64>, x_u: Vec<f64>, obj_constant: f64| NlProblemParts {
        minimize: true,
        objective: Expr::Var(0),
        obj_constant,
        constraints: vec![],
        x_l,
        x_u,
        x0: vec![0.0],
        g_l: vec![],
        g_u: vec![],
        var_names: vec![],
        con_names: vec![],
    };

    // Premise: the finite model builds.
    NlProblem::from_expressions(parts(vec![0.0], vec![1.0], 0.0)).expect("finite model builds");
    // `-inf` low / `+inf` high normalize, as in a file.
    let ok = NlProblem::from_expressions(parts(vec![f64::NEG_INFINITY], vec![f64::INFINITY], 0.0))
        .expect("the sentinel said differently");
    assert_eq!((ok.x_l[0], ok.x_u[0]), (-1e19, 1e19));
    // Everything else is refused.
    for (x_l, x_u, c, what) in [
        (vec![f64::INFINITY], vec![1.0], 0.0, "+inf lower bound"),
        (vec![0.0], vec![f64::NEG_INFINITY], 0.0, "-inf upper bound"),
        (vec![f64::NAN], vec![1.0], 0.0, "NaN lower bound"),
        (vec![0.0], vec![f64::NAN], 0.0, "NaN upper bound"),
        (vec![0.0], vec![1.0], f64::NAN, "NaN objective constant"),
        (
            vec![0.0],
            vec![1.0],
            f64::INFINITY,
            "inf objective constant",
        ),
    ] {
        let err = NlProblem::from_expressions(parts(x_l, x_u, c))
            .expect_err(&format!("{what} must be refused"));
        assert!(err.contains("invalid number"), "{what}: got {err}");
    }
}
