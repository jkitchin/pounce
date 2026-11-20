//! gh#1211 — the tail second-order correction (`tail_soc`) stays out of the
//! restoration phase.
//!
//! The restoration sub-IPM is built from a clone of the outer algorithm
//! builder, so it inherits every line-search option, `tail_soc` included.
//! The correction targets the last barrier problem of the *user's* NLP;
//! restoration ends on the outer filter's acceptance instead, and the inner
//! IPM's own monotone μ reaches its floor routinely. So
//! `pounce_restoration::resto_inner_solver` switches the option off for the
//! inner IPM, next to the `theta_max_row_scale_kappa` and
//! `theta_max_adaptive_trigger` opt-outs.
//!
//! The fixture is the infeasible circle–parabola model (`t = 1.5`): the main
//! phase never reaches its own μ floor at a feasible point, and the solve
//! spends most of its iterations in restoration, so it is the branch the
//! opt-out acts on. Measured with the opt-out removed: corrections fire
//! inside restoration, change the point it hands back, and the
//! infeasibility verdict takes 777 iterations instead of 603. With it, the
//! iteration log — restoration rows included — is identical with the option
//! on and off, which is what this test pins.
//!
//! Every second-opinion rung is switched off so the log is the primary
//! solve's alone.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// `(iteration rows, EXIT line)` of one solve.
fn solve(tail_soc: &str) -> (Vec<String>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture("circle_parabola_t1p5.nl"))
        .args([
            "feral_infeasibility_scaling_retry=no",
            "infeasibility_mu_strategy_retry=no",
            "infeasibility_perturbed_start_retry=no",
            "feral_increase_quality_retry=no",
            "mu_strategy=monotone",
            &format!("tail_soc={tail_soc}"),
        ])
        .output()
        .expect("spawn pounce");
    let text = String::from_utf8_lossy(&out.stdout);
    // An iteration row starts with the iteration number (restoration rows
    // carry an `r` suffix on it).
    let rows: Vec<String> = text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.split_whitespace().next().is_some_and(|w| {
                let w = w.trim_end_matches('r');
                !w.is_empty() && w.chars().all(|c| c.is_ascii_digit())
            })
        })
        .map(str::to_owned)
        .collect();
    let exit = text
        .lines()
        .rfind(|l| l.starts_with("EXIT:"))
        .unwrap_or_else(|| panic!("no EXIT line:\n{text}"))
        .to_owned();
    (rows, exit)
}

#[test]
fn tail_soc_does_not_act_inside_restoration() {
    let (off, exit_off) = solve("no");
    let (on, exit_on) = solve("yes");
    assert!(
        off.iter().any(|r| r
            .split_whitespace()
            .next()
            .is_some_and(|w| w.ends_with('r'))),
        "the fixture must reach restoration, or this test is not about the opt-out"
    );
    assert_eq!(exit_off, exit_on);
    assert_eq!(off.len(), on.len(), "iteration count moved with tail_soc");
    for (a, b) in off.iter().zip(&on) {
        assert_eq!(a, b, "iteration log diverged with tail_soc on");
    }
}
