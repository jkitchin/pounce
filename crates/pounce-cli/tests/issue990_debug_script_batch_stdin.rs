//! gh#990 item 14: `--debug-script` must not block on an open, silent stdin.
//!
//! A Jupyter kernel (or a CI step, or `subprocess.Popen(stdin=PIPE)`) hands
//! the child a pipe that is never written to and never closed. The script's
//! `continue` resumes to the next pause, which used to read a command from
//! that pipe forever. With a script and a non-terminal stdin the end of the
//! script now means "let it run", exactly as EOF on `/dev/null` always did.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn pounce_exe() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pounce"));
    p.set_extension(std::env::consts::EXE_EXTENSION);
    p.to_string_lossy().into_owned()
}

#[test]
fn an_exhausted_script_with_an_open_stdin_pipe_finishes_the_solve() {
    let script = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("issue990_batch.pdbg");
    std::fs::write(&script, "print x\ncontinue\n").expect("write debug script");
    let mut child = Command::new(pounce_exe())
        .arg(format!(
            "{}/tests/fixtures/airport.nl",
            env!("CARGO_MANIFEST_DIR")
        ))
        .arg("--no-sol")
        .arg("solver_selection=nlp")
        .arg("--debug-script")
        .arg(&script)
        // Left open and never written to: the Jupyter-kernel shape.
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pounce");
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("pounce still running 60 s after its script ended: it is blocked on stdin");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "unexpected exit {status:?}");
}
