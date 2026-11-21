//! `wasm_iterate_callback*` configures a call only the WebAssembly module
//! makes, so the CLI refuses a non-default value on every route. The NLP
//! route reaches `optimize_tnlp`'s guard; an LP or convex QP is routed to
//! `pounce-convex` before that and solved with the option doing nothing until
//! the CLI checked it before routing (PR #1215 review).

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn run(model: &str, opt: &str) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(model))
        .arg("--no-sol")
        .arg(opt)
        .output()
        .expect("run pounce");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned() + &String::from_utf8_lossy(&out.stdout),
    )
}

/// One model per route: the convex QP and LP interior point, and the NLP arm.
const MODELS: [&str; 3] = ["convex_qp_qscfxm1.nl", "lp_degen2.nl", "ssebnln.nl"];

#[test]
fn a_non_default_value_is_refused_on_every_route() {
    for model in MODELS {
        for opt in ["wasm_iterate_callback=yes", "wasm_iterate_callback_every=5"] {
            let (ok, text) = run(model, opt);
            assert!(!ok, "{model} {opt} must be refused:\n{text}");
            let name = opt.split('=').next().unwrap();
            assert!(
                text.contains(&format!("`{name}` configures the per-iteration host call")),
                "{model} {opt}:\n{text}"
            );
        }
    }
}

#[test]
fn the_defaults_are_not_refused() {
    let (ok, text) = run("lp_degen2.nl", "wasm_iterate_callback=no");
    assert!(ok, "{text}");
    assert!(!text.contains("per-iteration host call"), "{text}");
}
