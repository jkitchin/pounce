//! The corrector and the activity classification over Ipopt's result.
//!
//! Two models in `fixtures/cross_engine/`, each solved twice: by pounce, and
//! by Ipopt 3.13.2 with MA57 through its AMPL interface, both with
//! `bound_relax_factor = 0`. `cross_engine_scaled.nl` is `cross_engine.nl`
//! with the objective multiplied by 1000. Its gradient at the starting point
//! is 6180, so both solvers multiply its objective by `100/6180` and iterate
//! on bound multipliers that much smaller than the ones the `.sol` reports.
//!
//! pounce's run is its own: it classifies its converged point and corrects
//! its own parametric step. Ipopt's run goes through an engine outside
//! pounce, built from Ipopt's final iterate as the `.sol` file reports it,
//! the objective scaling factor and the barrier parameter of the last
//! iteration that Ipopt printed, and the model's own callbacks from the same
//! `.nl`. The engine iterates in Ipopt's frame, with the objective and the
//! reported multipliers multiplied by that factor, which is the frame the
//! printed barrier parameter belongs to. It fills in the inequality slacks
//! and their bound multipliers, which a `.sol` does not carry, factors its
//! own KKT matrix with a dense LU, takes its own parametric step through that
//! factorization, and runs the corrector and the classification over its own
//! iterate.
//!
//! The two runs agree to the accuracy the two solves agree to, which is
//! what the assertions below hold them to.
//!
//! The `.sol` files are committed, so this runs anywhere. With
//! `POUNCE_TEST_IPOPT` naming an Ipopt that has MA57,
//! `ipopt_still_produces_the_committed_results` re-solves each `.nl` and
//! checks the committed files are still its answer.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nl::nl_reader::{NlTnlp, read_nl_file};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_sensitivity::Solver;

mod common;
use common::outside_engine::{Bounds, OutsideEngine, correct_outside, parametric_step_outside};
use common::sol_file::{SolFile, ipopt_final_mu, ipopt_objective_scaling};

/// The model as written.
const UNSCALED: &str = "cross_engine";
/// The model with its objective multiplied by 1000.
const SCALED: &str = "cross_engine_scaled";

/// `pin`, the equality row holding the parameter, is the second row of both
/// `.row` files.
const PIN: Index = 1;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cross_engine")
        .join(name)
}

fn read_fixture(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn model(stem: &str) -> Rc<RefCell<dyn TNLP>> {
    let nl = read_nl_file(&fixture(&format!("{stem}.nl"))).expect("the fixture .nl parses");
    Rc::new(RefCell::new(NlTnlp::new(nl)))
}

/// pounce's own solve of the model, with Ipopt's options where they matter.
fn pounce_solved(stem: &str) -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let mut s = Solver::new(app, model(stem));
    let st = s.solve();
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{stem}: pounce's solve: {st:?}"
    );
    s
}

/// The engine outside pounce over Ipopt's committed result, in Ipopt's own
/// frame.
fn ipopt_engine(stem: &str) -> OutsideEngine {
    let sol = read_fixture(&format!("{stem}.sol"));
    let out = read_fixture(&format!("{stem}.ipopt.txt"));
    let mu = ipopt_final_mu(&out).expect("Ipopt printed its barrier parameter");
    let df = ipopt_objective_scaling(&out).expect("Ipopt printed its objective scaling");
    let reported = SolFile::parse(&sol)
        .expect("the .sol parses")
        .ipopt_iterate(mu);
    let m = model(stem);
    let declared = Bounds::declared(&mut *m.borrow_mut());
    OutsideEngine::with_objective_factor(m, &reported, &declared, &declared, df)
}

/// Ipopt and pounce reach the same point here: pounce ports Ipopt's
/// algorithm, and on these models the two iterate identically, so the primal
/// points agree to the last digit and the barrier parameters to the last two.
/// The comparisons below are therefore of the analyses, not of the solves.
fn ipopts_result_is_pounces_point(stem: &str) {
    let s = pounce_solved(stem);
    let eng = ipopt_engine(stem);
    let x = &s.converged().expect("converged").x;
    let n = x.len();
    for (k, (&a, &b)) in x.iter().zip(&eng.converged()[..n]).enumerate() {
        assert!(
            (a - b).abs() <= 1e-15 * a.abs().max(1.0),
            "{stem}: x[{k}]: pounce {a:e}, Ipopt {b:e}"
        );
    }
    let mu = s.classify_activity().expect("classification").mu;
    assert!(
        (mu - eng.mu()).abs() <= 1e-14 * mu,
        "{stem}: mu: pounce {mu:e}, Ipopt {:e}",
        eng.mu()
    );
}

#[test]
fn ipopts_result_is_pounces_point_on_the_unscaled_model() {
    ipopts_result_is_pounces_point(UNSCALED);
}

#[test]
fn ipopts_result_is_pounces_point_when_the_objective_is_scaled() {
    ipopts_result_is_pounces_point(SCALED);
}

/// The classification over Ipopt's result is pounce's. The statuses and the
/// flags agree exactly. The variables' ratios agree to `1e-15` on the
/// unscaled model and `3e-13` on the scaled one, and the active row's to
/// `3e-9` and `4e-14`. The inactive two-sided row `band` agrees to `1.5%` and
/// `0.08%`: a `.sol` carries one multiplier for it, the engine puts the
/// barrier's own `μ/s` on its inactive side, and the solve was centred to
/// that degree, which the comparison allows for an inactive row.
fn the_classification_over_ipopts_result_is_pounces(stem: &str) {
    let s = pounce_solved(stem);
    let eng = ipopt_engine(stem);
    let a = s.classify_activity().expect("pounce's classification");
    let b = pounce_sens_core::activity::compute(&eng).expect("the classification over Ipopt");
    let label = format!("Ipopt, {stem}");
    common::compare::assert_same_activity_within(&label, &a, &b, &eng.allowance());
    let vars: Vec<usize> = (0..a.var_status.len()).collect();
    let rows: Vec<usize> = (0..a.row_status.len()).collect();
    common::outside_engine::same_reduced_activity(&label, &s, &eng, &vars);
    common::outside_engine::same_reduced_row_activity(&label, &s, &eng, &rows);
}

#[test]
fn the_classification_over_ipopts_result_is_pounces_on_the_unscaled_model() {
    the_classification_over_ipopts_result_is_pounces(UNSCALED);
}

#[test]
fn the_classification_over_ipopts_result_is_pounces_when_the_objective_is_scaled() {
    the_classification_over_ipopts_result_is_pounces(SCALED);
}

/// Each engine takes its own parametric step through its own factorization
/// and corrects it, and the corrected primal points agree.
///
/// At `delta = 0.02` both corrections reach roundoff, and the points agree
/// to `2.2e-16` on both models, held here to [`SOLVED`]. At `0.2` and `-0.3`
/// the corrector stops on its second iteration, which fails to improve on
/// the first, with residuals of `3.2e-4` and `0.23` on the unscaled model and
/// `1.2e-2` and `3.1` on the scaled one. The point it returns is one Newton
/// step from the predicted point, so it carries the difference between the
/// two engines' predicted points, at most `2.4e-11`, through that step. On
/// the scaled model at `-0.3` the step amplifies it to `3.0e-8`, and
/// [`STOPPED`] holds the points to that. Everything discrete in the reports
/// is compared exactly at every perturbation.
const SOLVED: Number = 1e-12;
/// See [`SOLVED`].
const STOPPED: Number = 1e-7;

fn the_correction_over_ipopts_result_is_pounces(stem: &str) {
    let s = pounce_solved(stem);
    let eng = ipopt_engine(stem);
    let x = s.converged().expect("converged").x.clone();
    let n = x.len();
    let xi = eng.converged()[..n].to_vec();
    for delta in [0.02, 0.2, -0.3] {
        let step = s
            .parametric_step_full(&[PIN], &[delta])
            .expect("pounce's step");
        let (out, rep) = s
            .correct_step(&[PIN], &[delta], &step, 8)
            .expect("pounce's corrector");
        let step_i = parametric_step_outside(&eng, &[PIN], &[delta]);
        let (out_i, rep_i) =
            correct_outside(&eng, &[PIN], &[delta], &step_i, 8).expect("the corrector over Ipopt");
        let solved = rep.residual.max(rep_i.residual) <= common::compare::RESIDUAL_ATOL;
        let tol = if solved { SOLVED } else { STOPPED };
        for k in 0..n {
            let (a, b) = (x[k] + out[k], xi[k] + out_i[k]);
            assert!(
                (a - b).abs() <= tol,
                "{stem}, delta={delta}: corrected x[{k}] is {a:e} over pounce and {b:e} over Ipopt",
            );
        }
        let label = format!("{stem}, delta={delta}");
        assert_eq!(
            (rep.released, rep.pinned),
            (rep_i.released, rep_i.pinned),
            "{label}: the active-set decision differs
  pounce {rep:?}
  Ipopt  {rep_i:?}",
        );
        // Each engine starts from its own parametric step, so the initial
        // residuals differ with the steps. The final ones are compared.
        common::compare::assert_same_iterations(&label, &rep, &rep_i);
        assert!(
            common::compare::close(
                rep.residual,
                rep_i.residual,
                common::compare::RESIDUAL_RTOL,
                common::compare::RESIDUAL_ATOL,
            ),
            "{label}: residual {:e} over pounce, {:e} over Ipopt",
            rep.residual,
            rep_i.residual,
        );
    }
}

#[test]
fn the_correction_over_ipopts_result_is_pounces_on_the_unscaled_model() {
    the_correction_over_ipopts_result_is_pounces(UNSCALED);
}

#[test]
fn the_correction_over_ipopts_result_is_pounces_when_the_objective_is_scaled() {
    the_correction_over_ipopts_result_is_pounces(SCALED);
}

/// Re-solves each committed `.nl` with the Ipopt `POUNCE_TEST_IPOPT` names
/// and checks the committed `.sol`, objective scaling and barrier parameter
/// are still its answer, so the fixtures cannot drift from what they say
/// they are.
#[test]
#[ignore = "needs an Ipopt with MA57: set POUNCE_TEST_IPOPT to it and run with --ignored"]
fn ipopt_still_produces_the_committed_results() {
    let ipopt = std::env::var("POUNCE_TEST_IPOPT").expect("POUNCE_TEST_IPOPT names an ipopt");
    for stem in [UNSCALED, SCALED] {
        let dir = std::env::temp_dir().join(format!("pounce-{stem}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let nl = format!("{stem}.nl");
        std::fs::copy(fixture(&nl), dir.join(&nl)).unwrap();
        let run = std::process::Command::new(&ipopt)
            .current_dir(&dir)
            .args([
                nl.as_str(),
                "-AMPL",
                "linear_solver=ma57",
                "bound_relax_factor=0",
                "print_level=8",
            ])
            .output()
            .expect("ipopt runs");
        let printed = String::from_utf8_lossy(&run.stdout).replace('\r', "");
        assert!(
            printed.contains("running with linear solver ma57"),
            "{stem}: Ipopt did not run MA57"
        );
        let committed_out = read_fixture(&format!("{stem}.ipopt.txt"));
        assert_eq!(
            ipopt_final_mu(&printed),
            ipopt_final_mu(&committed_out),
            "{stem}: the barrier parameter moved"
        );
        assert_eq!(
            ipopt_objective_scaling(&printed),
            ipopt_objective_scaling(&committed_out),
            "{stem}: the objective scaling moved"
        );
        let fresh =
            SolFile::parse(&std::fs::read_to_string(dir.join(format!("{stem}.sol"))).unwrap())
                .unwrap();
        let committed = SolFile::parse(&read_fixture(&format!("{stem}.sol"))).unwrap();
        for (what, a, b) in [
            ("primals", &fresh.primals, &committed.primals),
            ("duals", &fresh.duals, &committed.duals),
        ] {
            assert_eq!(a.len(), b.len(), "{stem}: {what}: lengths");
            for (k, (&u, &v)) in a.iter().zip(b).enumerate() {
                assert!(
                    (u - v).abs() <= 1e-12 * u.abs().max(1.0),
                    "{stem}: {what}[{k}]: {u:e} vs {v:e}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
