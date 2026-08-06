//! Issue #495 end to end: the bound multiplier a collapsed reduced box
//! hides has to reach the `.sol` suffix blocks, because that is where
//! Pyomo (`model.ipopt_zL_out`) and AMPL (`.rc`) read it from.
//!
//! The fixture is the issue's own minimal model:
//!
//! ```text
//!   min (x0−4)⁴ + (x1−4)⁴   s.t.  x0 − x1 = 0,  x0 ∈ [1,5],  x1 ∈ [−5,1]
//! ```
//!
//! Phase 6 folds `x0` onto `x1` and carries `x0 ≥ 1` across, which leaves
//! `x1` with the reduced box `[1,1]` — a fixed variable, which the solver
//! drops, so nothing in the reduced problem can produce a multiplier for
//! it. The optimum is `x = (1,1)` with `∇f = (−108, −108)`, so `λ = 108`
//! and a bound multiplier of `216` on `x1`'s upper bound close
//! stationarity. Before the fix both suffix blocks came back empty.
//!
//! `presolve_bound_tightening=no` is not incidental: Phase 1 would
//! otherwise pin both columns from the same intersection and the model
//! would never reach Phase 6 at all.

use std::path::PathBuf;
use std::process::Command;

/// `(row duals, primals, zL entries, zU entries)` read out of a `.sol`.
struct Sol {
    duals: Vec<f64>,
    x: Vec<f64>,
    z_l: Vec<(usize, f64)>,
    z_u: Vec<(usize, f64)>,
}

fn run(tag: &str, opts: &[&str]) -> Sol {
    let dir = std::env::temp_dir().join(format!("pounce_lin_eq_collapsed_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixture.push("tests/fixtures/linear_eq_collapsed_box.nl");
    std::fs::copy(&fixture, dir.join("m.nl")).expect("copy fixture");

    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.current_dir(&dir).arg("m").arg("-AMPL");
    for o in opts {
        cmd.arg(o);
    }
    let out = cmd.output().expect("run pounce");
    let sol = std::fs::read_to_string(dir.join("m.sol")).unwrap_or_else(|e| {
        panic!(
            "no .sol: {e}\n{}",
            String::from_utf8_lossy(&out.stdout).into_owned()
        )
    });

    // Body: the `Options` preamble, then 1 dual and 2 primals, then the
    // suffix blocks.
    let body: Vec<f64> = sol
        .lines()
        .take_while(|l| !l.starts_with("objno"))
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(body.len() >= 3, "short .sol body {body:?}\n{sol}");
    let (mut z_l, mut z_u) = (Vec::new(), Vec::new());
    let mut into: Option<&mut Vec<(usize, f64)>> = None;
    for line in sol.lines().skip_while(|l| !l.starts_with("objno")) {
        match line.trim() {
            "ipopt_zL_out" => into = Some(&mut z_l),
            "ipopt_zU_out" => into = Some(&mut z_u),
            other => {
                let mut it = other.split_whitespace();
                if let (Some(Ok(i)), Some(Ok(v)), Some(slot)) = (
                    it.next().map(str::parse::<usize>),
                    it.next().map(str::parse::<f64>),
                    into.as_deref_mut(),
                ) {
                    slot.push((i, v));
                }
            }
        }
    }
    Sol {
        duals: body[body.len() - 3..body.len() - 2].to_vec(),
        x: body[body.len() - 2..].to_vec(),
        z_l,
        z_u,
    }
}

/// The `.sol` writer records `z_u` with the sign AMPL expects, so compare
/// magnitudes against the bare solve rather than hard-coding a convention
/// this test does not own.
fn magnitude(entries: &[(usize, f64)], j: usize) -> f64 {
    entries
        .iter()
        .find(|(i, _)| *i == j)
        .map(|(_, v)| v.abs())
        .unwrap_or(0.0)
}

#[test]
fn a_collapsed_reduced_box_still_writes_its_bound_multiplier_suffix() {
    let reduced = run(
        "reduced",
        &[
            "presolve=yes",
            "presolve_linear_eq_reduction=yes",
            "presolve_bound_tightening=no",
        ],
    );
    let bare = run("bare", &[]);

    assert!(
        (reduced.x[0] - 1.0).abs() < 1e-6 && (reduced.x[1] - 1.0).abs() < 1e-6,
        "primal moved: {:?}",
        reduced.x
    );
    assert!(
        (reduced.duals[0] - bare.duals[0]).abs() < 1.0,
        "row dual {} vs bare {}",
        reduced.duals[0],
        bare.duals[0]
    );

    // The one the reduced problem could not produce: 216 on x1's upper
    // bound. The bare solve reports the same thing to interior-point slack.
    let got = magnitude(&reduced.z_u, 1);
    assert!(
        (got - 216.0).abs() < 1.0,
        "expected |z_u[1]| ≈ 216, got {got} (bare solve: {}); zL = {:?}, zU = {:?}",
        magnitude(&bare.z_u, 1),
        reduced.z_l,
        reduced.z_u
    );
    assert!(
        (got - magnitude(&bare.z_u, 1)).abs() < 1.0,
        "diverged from the bare solve: {got} vs {}",
        magnitude(&bare.z_u, 1)
    );

    // Complementarity: x0 sits on its lower bound and x1 on its upper, so
    // no multiplier may appear on either slack side.
    assert!(
        magnitude(&reduced.z_u, 0) < 1e-4,
        "multiplier on x0's slack upper bound: {:?}",
        reduced.z_u
    );
    assert!(
        magnitude(&reduced.z_l, 1) < 1e-4,
        "multiplier on x1's slack lower bound: {:?}",
        reduced.z_l
    );
}
