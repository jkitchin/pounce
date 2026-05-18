//! Integration test for the `--dump kkt:...` surface.
//!
//! Solves HS071 with a [`DiagnosticsState`] installed on the
//! application asking for KKT dumps over iters 1-2, then verifies:
//!
//! * the dump directory tree has the expected `iter_NNN/kkt_solve_001.jsonl`
//!   files (and only those iters),
//! * each dumped record is parseable as one JSONL line with the
//!   expected top-level fields,
//! * the top-level `manifest.json` helper works.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::diagnostics::{
    DiagCategory, DiagnosticsConfig, DiagnosticsState, IterSpec,
};
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::fs;
use std::rc::Rc;

#[derive(Default)]
struct Hs071;

impl TNLP for Hs071 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo { n: 4, m: 2, nnz_jac_g: 8, nnz_h_lag: 10, index_style: IndexStyle::C })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[1.0; 4]);
        b.x_u.copy_from_slice(&[5.0; 4]);
        b.g_l.copy_from_slice(&[25.0, 40.0]);
        b.g_u.copy_from_slice(&[2.0e19, 40.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        true
    }
    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = x[1] * x[2] * x[3];
                values[1] = x[0] * x[2] * x[3];
                values[2] = x[0] * x[1] * x[3];
                values[3] = x[0] * x[1] * x[2];
                values[4] = 2.0 * x[0];
                values[5] = 2.0 * x[1];
                values[6] = 2.0 * x[2];
                values[7] = 2.0 * x[3];
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
                jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let lam = lambda.expect("eval_h(Values) without lambda");
                let of = obj_factor;
                let l0 = lam[0];
                let l1 = lam[1];
                values[0] = of * (2.0 * x[3]) + l1 * 2.0;
                values[1] = of * x[3] + l0 * (x[2] * x[3]);
                values[2] = l1 * 2.0;
                values[3] = of * x[3] + l0 * (x[1] * x[3]);
                values[4] = l0 * (x[0] * x[3]);
                values[5] = l1 * 2.0;
                values[6] = of * (2.0 * x[0] + x[1] + x[2]) + l0 * (x[1] * x[2]);
                values[7] = of * x[0] + l0 * (x[0] * x[2]);
                values[8] = of * x[0] + l0 * (x[0] * x[1]);
                values[9] = l1 * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "pounce-diag-it-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn kkt_dump_produces_per_iter_files_and_manifest() {
    let dump_dir = tempdir("kkt");

    let cfg = DiagnosticsConfig::new(dump_dir.clone())
        .with_category(DiagCategory::Kkt, IterSpec::Range(Some(1), Some(2)));
    let diag = Rc::new(DiagnosticsState::new(cfg).unwrap());

    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.set_diagnostics(Rc::clone(&diag));

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071));
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    // Iters 1 and 2 should each have at least one kkt dump.
    for iter in [1, 2] {
        let dir = dump_dir.join(format!("iter_{iter:03}"));
        assert!(dir.is_dir(), "missing dump dir for iter {iter}: {dir:?}");
        let solve = dir.join("kkt_solve_001.jsonl");
        assert!(solve.is_file(), "missing dump file: {solve:?}");
        let body = fs::read_to_string(&solve).unwrap();
        assert!(body.starts_with('{') && body.contains("\"n\":"), "bad record: {body}");
        assert!(body.contains("\"vals\":["), "missing vals field");
        assert!(body.contains("\"sol\":["), "missing sol field");
        assert!(body.ends_with("]}\n"), "missing terminator: {body}");
    }

    // Out-of-range iter should not have produced a kkt file (the
    // directory only gets created on demand by `iter_dir()`).
    let out_of_range = dump_dir.join("iter_000");
    if out_of_range.exists() {
        let entries: Vec<_> = fs::read_dir(&out_of_range).unwrap().collect();
        assert!(
            entries.is_empty(),
            "iter_000 should not contain kkt files (range was 1-2)",
        );
    }

    // The top-level writer helper drops files at `dump_dir`.
    diag.write_top_level("manifest.json", "{\"hello\":\"world\"}\n").unwrap();
    let manifest = fs::read_to_string(dump_dir.join("manifest.json")).unwrap();
    assert!(manifest.contains("\"hello\":\"world\""));

    fs::remove_dir_all(&dump_dir).ok();
}
