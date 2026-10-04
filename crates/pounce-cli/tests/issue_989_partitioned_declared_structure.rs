//! gh #989 item 5: `hessian_approximation=partitioned` took 16 / 28 / 86
//! iterations against the exact Hessian's 12 / 16 / 18 as a Radau-collocated
//! batch reactor was refined (N = 25 / 100 / 400 finite elements),
//! `partitioned_update_type=bfgs` hit the iteration cap at every size, and
//! `partitioned_elements=blocks` took 80 / 163 / 225 and then failed.
//!
//! Root cause: every element block was updated with a *dense* formula (SR1,
//! damped BFGS) and seeded with `γ I`, while the true element Hessian is
//! sparse inside its support. A collocation row's declared group is a star —
//! the control coupled to each state at the same point, no state–state and no
//! state-diagonal entries — so the dense update wrote curvature into entries
//! that are structurally zero, and those entries, weighted by multipliers of
//! either sign, gave the assembled `W` the wrong inertia. The IPM then ran on a
//! `δ_w` that decayed by 1/3 per iteration, which is the mesh growth.
//! Elements whose declared pattern is incomplete now take the
//! pattern-constrained (Toint) secant update from `B = 0`; on a star that
//! determines the element Hessian from one pair. With a declared pattern,
//! `blocks` takes the pattern's connected components as its blocks instead of
//! contiguous index ranges (an AMPL `.nl` orders nonlinear variables first, so
//! contiguous ranges are not stages).
//!
//! Fixtures: the issue's model (discopt `DAEBuilder`, Radau, 3 collocation
//! points) exported with `to_nl()` at N = 25 and N = 50, starting from the
//! issue's analytic `T = 330 K` profile. Measured after the fix, iterations
//! exact / partitioned / bfgs / blocks: N = 25 12/13/13/13, N = 50 17/15/15/15,
//! N = 100 16/15/15/15, N = 400 18/21/21/21.

use std::process::Command;

fn iterations(n: u32, extra: &[&str]) -> (u32, bool) {
    let path = format!(
        "{}/tests/data/batch_reactor_radau_n{n}.nl",
        env!("CARGO_MANIFEST_DIR")
    );
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&path)
        .args(["--no-sol", "--no-options-file", "max_iter=300"])
        .args(extra)
        .output()
        .expect("run pounce");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let it = text
        .lines()
        .find(|l| l.starts_with("Number of Iterations"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no iteration count in:\n{text}"));
    (it, text.contains("EXIT: Optimal Solution Found"))
}

const VARIANTS: [(&str, &[&str]); 3] = [
    ("partitioned", &["hessian_approximation=partitioned"]),
    (
        "partitioned+bfgs",
        &[
            "hessian_approximation=partitioned",
            "partitioned_update_type=bfgs",
        ],
    ),
    (
        "partitioned+blocks",
        &[
            "hessian_approximation=partitioned",
            "partitioned_elements=blocks",
        ],
    ),
];

/// Every partitioned variant converges and stays within 1.5x of the exact
/// Hessian at both mesh sizes. Before the fix: partitioned 14 and 22 against
/// exact 12 and 17, bfgs at the cap, blocks 88 and 122.
#[test]
fn every_partitioned_variant_tracks_the_exact_hessian_on_both_meshes() {
    for n in [25, 50] {
        let (exact, ok) = iterations(n, &["hessian_approximation=exact"]);
        assert!(ok, "exact must converge at N={n}");
        for (name, opts) in VARIANTS {
            let (it, ok) = iterations(n, opts);
            assert!(ok, "{name} must converge at N={n} ({it} iterations)");
            assert!(
                it * 10 <= exact * 15,
                "{name} ({it}) must stay within 1.5x of the exact Hessian ({exact}) at N={n}"
            );
        }
    }
}

/// Doubling the mesh must not grow the partitioned iteration count faster
/// than it grows the exact one (before the fix: 14 -> 22 against 12 -> 17,
/// and 14 -> 54 -> 74 out to N = 400).
#[test]
fn partitioned_iterations_do_not_grow_with_the_mesh() {
    let (e25, _) = iterations(25, &["hessian_approximation=exact"]);
    let (e50, _) = iterations(50, &["hessian_approximation=exact"]);
    let (p25, _) = iterations(25, VARIANTS[0].1);
    let (p50, _) = iterations(50, VARIANTS[0].1);
    assert!(
        p50 <= p25 + (e50.saturating_sub(e25)) + 3,
        "partitioned grew {p25} -> {p50} while exact grew {e25} -> {e50}"
    );
}

/// The declared structure still beats dense per-row blocks
/// (`partitioned_structure=jacobian`).
#[test]
fn declared_structure_beats_dense_rows() {
    let (jac, ok_j) = iterations(
        50,
        &[
            "hessian_approximation=partitioned",
            "partitioned_structure=jacobian",
        ],
    );
    let (decl, ok_d) = iterations(50, VARIANTS[0].1);
    assert!(ok_j && ok_d, "both must converge");
    assert!(
        decl < jac,
        "declared structure ({decl}) must beat dense rows ({jac})"
    );
}

/// gh#989 item 5, remaining half: `partitioned_update_type=bfgs` with **no**
/// declared Hessian pattern (`partitioned_structure=jacobian`, also what a
/// Python problem without `hessian` gets). Damped BFGS forced every
/// constraint element positive definite while `y_j·∇²c_j` has either sign,
/// and the assembled `W` had the wrong inertia: the cap at N = 25 and 50.
/// Constraint elements now take SR1 under `bfgs` (only the objective element
/// is damped BFGS), so `bfgs` converges within 2x of `sr1` (measured: equal,
/// 16 and 23 iterations).
#[test]
fn bfgs_without_a_declared_pattern_converges_like_sr1() {
    for n in [25, 50] {
        let base = [
            "hessian_approximation=partitioned",
            "partitioned_structure=jacobian",
        ];
        let (sr1, ok_s) = iterations(n, &[base[0], base[1], "partitioned_update_type=sr1"]);
        let (bfgs, ok_b) = iterations(n, &[base[0], base[1], "partitioned_update_type=bfgs"]);
        assert!(ok_s, "sr1 must converge at N={n} ({sr1} iterations)");
        assert!(ok_b, "bfgs must converge at N={n} ({bfgs} iterations)");
        assert!(
            bfgs <= 2 * sr1,
            "N={n}: bfgs {bfgs} iterations against sr1 {sr1}"
        );
    }
}
