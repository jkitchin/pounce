//! Replay **real POUNCE KKT systems** through FERAL, RSLAB and (under
//! `--features ma57`) MA57.
//!
//! The KKT matrices are not synthesised and not read from a saved dump: the
//! example runs an actual POUNCE solve of an `.nl` model with a capturing
//! backend installed via `IpoptApplication::set_linear_backend_factory`, so
//! every system it replays is one the interior-point loop genuinely asked for,
//! complete with the negative-eigenvalue count the loop expected at that
//! iteration. That last field matters: a backend can produce a fine residual
//! and still be useless to POUNCE if it disagrees about the inertia.
//!
//! ```sh
//! cargo run -p pounce-rslab --release --example rslab_kkt_replay
//! cargo run -p pounce-rslab --release --example rslab_kkt_replay -- path/to/model.nl
//! ```
//!
//! The default model list is chosen, not convenient:
//!
//! * `eigena2` — the model pounce gh#540 was filed against. Its KKT
//!   factorizations have pivots at the working-precision floor, and FERAL
//!   returns 64 / 58 / 62 negatives across runs for the same expected 55, none
//!   of which LAPACK's exact count agrees with. This is the "correct inertia
//!   but not a measurement" case, and the one that decides whether RSLAB is
//!   any better.
//! * `eigenb2` — its sibling, same family, different conditioning.
//! * `deb7` — the largest fixture of any class in the corpus (813 variables),
//!   so the fill and timing rows have something to say.
//! * `convex_qp_qscfxm1` — QSCFXM1 itself, the Maros-Meszaros member pinned by
//!   `crates/pounce-cli/tests/issue_760_convex_bound_relax_magnitude.rs`,
//!   degenerate and the case where relaxing the box costs iterations.
//! * `airport`, `lp_degen2` — a mid-size NLP and a degenerate LP, for the
//!   rank-deficient-Jacobian shape.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use pounce_algorithm::alg_builder::LinearSolverChoice;
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_linsol::{
    EMatrixFormat, ESymSolverStatus, FactorPattern, SparseSymLinearSolverInterface,
};
use pounce_rslab::compare::{self, SymTriplet};

/// One KKT system as the IPM handed it over, plus what the IPM expected of it.
#[derive(Clone)]
struct CapturedKkt {
    matrix: SymTriplet,
    /// The right-hand side of the solve that followed the factorization.
    rhs: Vec<Number>,
    /// Whether the IPM asked for an inertia check on this factorization, and
    /// the count it wanted. This is the contract RSLAB has to satisfy — a
    /// solver that factors beautifully and disagrees here sends the IPM up the
    /// `δ_w` ladder for nothing.
    check_neg_evals: bool,
    expected_neg: Index,
    /// What the FERAL backend that ran the solve actually reported.
    feral_neg: Index,
    feral_status: ESymSolverStatus,
}

/// A backend that delegates everything to an inner backend and records the
/// systems that pass through it.
struct Capturing {
    inner: pounce_feral::FeralSolverInterface,
    sink: Arc<Mutex<Vec<CapturedKkt>>>,
    /// Keep every `stride`-th factorization. A solve of a few hundred
    /// iterations issues more systems than there is any point replaying, and
    /// they are highly correlated between neighbours.
    stride: usize,
    seen: usize,
    dim: Index,
    irn: Vec<Index>,
    jcn: Vec<Index>,
}

impl Capturing {
    fn new(sink: Arc<Mutex<Vec<CapturedKkt>>>, stride: usize) -> Self {
        Self {
            inner: pounce_feral::FeralSolverInterface::new(),
            sink,
            stride,
            seen: 0,
            dim: 0,
            irn: Vec::new(),
            jcn: Vec::new(),
        }
    }
}

impl SparseSymLinearSolverInterface for Capturing {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        self.dim = dim;
        self.irn = ia.to_vec();
        self.jcn = ja.to_vec();
        self.inner.initialize_structure(dim, nonzeros, ia, ja)
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        self.inner.values_array_mut()
    }

    fn multi_solve(
        &mut self,
        new_matrix: bool,
        ia: &[Index],
        ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Snapshot the values and the RHS *before* the solve overwrites them.
        let vals = self.inner.values_array_mut().to_vec();
        let rhs = rhs_vals.to_vec();
        let st = self.inner.multi_solve(
            new_matrix,
            ia,
            ja,
            nrhs,
            rhs_vals,
            check_neg_evals,
            number_of_neg_evals,
        );
        if new_matrix && nrhs == 1 {
            let keep = self.seen % self.stride == 0;
            self.seen += 1;
            if keep {
                if let Ok(mut g) = self.sink.lock() {
                    g.push(CapturedKkt {
                        matrix: SymTriplet {
                            n: self.dim,
                            irn: self.irn.clone(),
                            jcn: self.jcn.clone(),
                            vals,
                        },
                        rhs,
                        check_neg_evals,
                        expected_neg: number_of_neg_evals,
                        feral_neg: self.inner.number_of_neg_evals(),
                        feral_status: st,
                    });
                }
            }
        }
        st
    }

    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
    }
    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }
    fn multi_solve_matches_single_solve(&self, nrhs: usize) -> bool {
        self.inner.multi_solve_matches_single_solve(nrhs)
    }
    fn matrix_format(&self) -> EMatrixFormat {
        self.inner.matrix_format()
    }
    fn factor_pattern(&self, want_values: bool) -> Option<FactorPattern> {
        self.inner.factor_pattern(want_values)
    }
}

/// Solve `path` with a capturing FERAL backend and return the systems kept.
fn capture(path: &Path, stride: usize) -> Result<Vec<CapturedKkt>, String> {
    let prob = pounce_nl::nl_reader::read_nl_file(path)?;
    let tnlp: Rc<RefCell<dyn pounce_nlp::tnlp::TNLP>> =
        Rc::new(RefCell::new(pounce_nl::nl_reader::NlTnlp::try_new(prob)?));

    let sink: Arc<Mutex<Vec<CapturedKkt>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_for_factory = Arc::clone(&sink);

    let mut app = IpoptApplication::new();
    app.initialize().map_err(|e| format!("{e:?}"))?;
    app.initialize_with_options_str("print_level 0\nmax_iter 200\n")
        .map_err(|e| format!("{e:?}"))?;
    app.set_linear_backend_factory(Box::new(
        move |_choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            Box::new(Capturing::new(Arc::clone(&sink_for_factory), stride))
        },
    ));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    let captured = sink.lock().map_err(|_| "sink poisoned")?.clone();
    eprintln!(
        "  solve: {status:?}  iters={}  f*={:.8e}  captured {} KKT systems",
        stats.iteration_count,
        stats.final_objective,
        captured.len()
    );
    Ok(captured)
}

/// Summary across one model's replayed systems.
#[derive(Default)]
struct Tally {
    systems: usize,
    rslab_factor_ok: usize,
    feral_factor_ok: usize,
    /// Both factored and agreed on the negative count.
    inertia_agree: usize,
    /// Both factored and disagreed.
    inertia_disagree: usize,
    /// RSLAB's residual_ratio was the smaller of the two.
    rslab_better_residual: usize,
    feral_better_residual: usize,
    rslab_2x2: usize,
    feral_nnz_l: usize,
    rslab_nnz_l: usize,
    feral_factor_ms: f64,
    rslab_factor_ms: f64,
    feral_refactor_ms: f64,
    rslab_refactor_ms: f64,
    feral_solve_ms: f64,
    rslab_solve_ms: f64,
    rslab_unreliable: usize,
    /// Systems BOTH backends factored — the denominator for every paired
    /// figure below.
    paired: usize,
    /// Slots of RSLAB's `L` holding an exact zero, over the paired systems.
    rslab_nnz_l_zeros: usize,
    /// Worst `rslab / feral` ratio of POUNCE's residual metric.
    worst_residual_gap: f64,
    /// The scaling-neutral control (both backends unscaled).
    ns_paired: usize,
    ns_feral_ok: usize,
    ns_rslab_ok: usize,
    ns_feral_better: usize,
    ns_rslab_better: usize,
    ns_worst_gap: f64,
    /// The static-pivoting arm — the only one that factors a POUNCE KKT.
    rslab_sp_factor_ok: usize,
    rslab_sp_perturbed: usize,
    rslab_sp_2x2: usize,
    rslab_sp_inertia_agree: usize,
    rslab_sp_inertia_disagree: usize,
    rslab_sp_worst_residual: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<PathBuf> = if args.is_empty() {
        default_models()
    } else {
        args.iter().map(PathBuf::from).collect()
    };
    let stride: usize = std::env::var("RSLAB_REPLAY_STRIDE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!("# RSLAB vs FERAL on real POUNCE KKT systems");
    println!("# captured through IpoptApplication::set_linear_backend_factory");
    println!("# replay stride: every {stride}th factorization\n");

    for model in &models {
        let name = model
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| model.display().to_string());
        if !model.is_file() {
            eprintln!("== {name}: missing ({})", model.display());
            continue;
        }
        eprintln!("== {name}");
        let captured = match capture(model, stride) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  capture failed: {e}");
                continue;
            }
        };
        if captured.is_empty() {
            eprintln!("  no KKT systems captured");
            continue;
        }

        let mut tally = Tally::default();
        println!("## {name}  (n = {}, nnz = {})", captured[0].matrix.n, captured[0].matrix.nnz());
        println!("{}", compare::ROW_HEADER);

        for (k, cap) in captured.iter().enumerate() {
            let mut recs = compare::run_all(&cap.matrix, &cap.rhs);
            recs.extend(compare::run_scaling_neutral_pair(&cap.matrix, &cap.rhs));
            let feral = recs.iter().find(|r| r.solver == "feral");
            let rslab = recs.iter().find(|r| r.solver == "rslab");
            tally.systems += 1;

            // Print the first, the middle and the last system in full; the
            // rest are folded into the tally. A table of 200 rows is not a
            // result, it is a haystack.
            let verbose = k == 0 || k == captured.len() / 2 || k + 1 == captured.len();
            if verbose {
                println!(
                    "# system {k}: expected_neg={} (checked={}) feral_reported={} feral_status={:?}",
                    cap.expected_neg, cap.check_neg_evals, cap.feral_neg, cap.feral_status
                );
                // The outside number: dense eigenvalues of the same matrix, so
                // "feral and rslab agree" can be distinguished from "feral and
                // rslab are both right". Skipped above 400 -- it is O(n^3) per
                // sweep and the point is a spot check, not a sweep.
                if let Some((p, n, z, lo, hi)) = compare::dense_inertia_oracle(&cap.matrix, 400) {
                    println!(
                        "# oracle (dense Jacobi): inertia ({p},{n},{z})  min|lambda| {lo:.3e}  max|lambda| {hi:.3e}  kappa {:.3e}",
                        if lo > 0.0 { hi / lo } else { f64::INFINITY }
                    );
                }
                for r in &recs {
                    println!("{}", r.row());
                }
            }

            // The scaling-neutral control: does the residual gap survive with
            // equilibration off on both sides?
            if let (Some(f), Some(r)) = (
                recs.iter().find(|r| r.solver == "feral-ns"),
                recs.iter().find(|r| r.solver == "rslab-ns"),
            ) {
                if f.factor_status == ESymSolverStatus::Success
                    && r.factor_status == ESymSolverStatus::Success
                {
                    tally.ns_paired += 1;
                    match (f.residual_ratio, r.residual_ratio) {
                        (Some(a), Some(b)) if b < a => tally.ns_rslab_better += 1,
                        (Some(a), Some(b)) if a < b => tally.ns_feral_better += 1,
                        _ => {}
                    }
                    if let (Some(a), Some(b)) = (f.residual_ratio, r.residual_ratio) {
                        tally.ns_worst_gap = tally
                            .ns_worst_gap
                            .max(if a > 0.0 { b / a } else { f64::INFINITY });
                    }
                }
                if r.factor_status == ESymSolverStatus::Success {
                    tally.ns_rslab_ok += 1;
                }
                if f.factor_status == ESymSolverStatus::Success {
                    tally.ns_feral_ok += 1;
                }
            }
            let rslab_sp = recs.iter().find(|r| r.solver == "rslab-sp");
            if let Some(sp) = rslab_sp {
                if sp.factor_status == ESymSolverStatus::Success {
                    tally.rslab_sp_factor_ok += 1;
                    tally.rslab_sp_perturbed += sp.perturbed_pivots.unwrap_or(0);
                    tally.rslab_sp_2x2 += sp.two_by_two_pivots.unwrap_or(0);
                    if let (Some(f), Some(e)) = (feral.and_then(|f| f.negative_evals), sp.negative_evals) {
                        if f == e {
                            tally.rslab_sp_inertia_agree += 1;
                        } else {
                            tally.rslab_sp_inertia_disagree += 1;
                        }
                    }
                    if let Some(rr) = sp.residual_ratio {
                        tally.rslab_sp_worst_residual = tally.rslab_sp_worst_residual.max(rr);
                    }
                }
            }
            if let (Some(f), Some(r)) = (feral, rslab) {
                let f_ok = f.factor_status == ESymSolverStatus::Success;
                let r_ok = r.factor_status == ESymSolverStatus::Success;
                if f_ok {
                    tally.feral_factor_ok += 1;
                }
                if r_ok {
                    tally.rslab_factor_ok += 1;
                    tally.rslab_2x2 += r.two_by_two_pivots.unwrap_or(0);
                    if r.inertia_reliable == Some(false) {
                        tally.rslab_unreliable += 1;
                    }
                }
                // Fill and timing are accumulated ONLY over systems both
                // backends factored. Summing each over its own successes would
                // compare a 35-system FERAL total against a 9-system RSLAB one
                // and read as a 5x speed-up that is entirely the count.
                if f_ok && r_ok {
                    tally.paired += 1;
                    tally.feral_nnz_l += f.nnz_l.unwrap_or(0);
                    tally.rslab_nnz_l += r.nnz_l.unwrap_or(0);
                    tally.rslab_nnz_l_zeros += r.explicit_zeros_in_l.unwrap_or(0);
                    tally.feral_factor_ms += f.factor_ms;
                    tally.rslab_factor_ms += r.factor_ms;
                    tally.feral_refactor_ms += f.refactor_ms.unwrap_or(0.0);
                    tally.rslab_refactor_ms += r.refactor_ms.unwrap_or(0.0);
                    tally.feral_solve_ms += f.solve_ms.unwrap_or(0.0);
                    tally.rslab_solve_ms += r.solve_ms.unwrap_or(0.0);
                    if f.negative_evals == r.negative_evals {
                        tally.inertia_agree += 1;
                    } else {
                        tally.inertia_disagree += 1;
                        println!(
                            "# INERTIA DISAGREEMENT system {k}: feral {:?}, rslab {:?}, expected {}",
                            f.negative_evals, r.negative_evals, cap.expected_neg
                        );
                    }
                    match (f.residual_ratio, r.residual_ratio) {
                        (Some(a), Some(b)) if b < a => tally.rslab_better_residual += 1,
                        (Some(a), Some(b)) if a < b => tally.feral_better_residual += 1,
                        _ => {}
                    }
                    if let (Some(a), Some(b)) = (f.residual_ratio, r.residual_ratio) {
                        tally.worst_residual_gap = tally
                            .worst_residual_gap
                            .max(if a > 0.0 { b / a } else { f64::INFINITY });
                    }
                }
            }
        }

        println!("\n### {name} summary over {} systems", tally.systems);
        println!(
            "  factor ok            feral {:>4}   rslab {:>4}   rslab-sp {:>4}",
            tally.feral_factor_ok, tally.rslab_factor_ok, tally.rslab_sp_factor_ok
        );
        println!(
            "  rslab-sp vs feral inertia agree/disagree  {} / {}   perturbed pivots (total) {}   2x2 (total) {}",
            tally.rslab_sp_inertia_agree,
            tally.rslab_sp_inertia_disagree,
            tally.rslab_sp_perturbed,
            tally.rslab_sp_2x2
        );
        println!(
            "  rslab-sp worst residual_ratio            {:.3e}",
            tally.rslab_sp_worst_residual
        );
        println!(
            "  scaling-neutral control (both unscaled): factor ok feral {} rslab {}; \
             over {} paired, smaller residual feral {} rslab {}, worst rslab/feral {:.2e}",
            tally.ns_feral_ok,
            tally.ns_rslab_ok,
            tally.ns_paired,
            tally.ns_feral_better,
            tally.ns_rslab_better,
            tally.ns_worst_gap
        );
        println!("  2x2 pivots (rslab, total)          {}", tally.rslab_2x2);
        println!("  rslab inertia flagged unreliable   {}", tally.rslab_unreliable);
        println!(
            "  -- paired comparison over the {} systems BOTH factored --",
            tally.paired
        );
        if tally.paired == 0 {
            println!("  (none: nothing to compare)");
        } else {
            println!(
                "  inertia agree/disagree            {:>4} / {}",
                tally.inertia_agree, tally.inertia_disagree
            );
            println!(
                "  smaller residual_ratio  feral {:>4}   rslab {:>4}   worst rslab/feral ratio {:.2e}",
                tally.feral_better_residual, tally.rslab_better_residual, tally.worst_residual_gap
            );
            println!(
                "  structural nnz(L)    feral {:>10}   rslab {:>10}  (of which exactly zero: {})",
                tally.feral_nnz_l, tally.rslab_nnz_l, tally.rslab_nnz_l_zeros
            );
            println!(
                "  total ms  factor  feral {:>9.2}  rslab {:>9.2}",
                tally.feral_factor_ms, tally.rslab_factor_ms
            );
            println!(
                "            refac   feral {:>9.2}  rslab {:>9.2}",
                tally.feral_refactor_ms, tally.rslab_refactor_ms
            );
            println!(
                "            solve   feral {:>9.2}  rslab {:>9.2}",
                tally.feral_solve_ms, tally.rslab_solve_ms
            );
        }
        println!();
    }
}

fn default_models() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("crates/pounce-cli/tests/fixtures");
    [
        "eigena2",
        "eigenb2",
        "deb7",
        "convex_qp_qscfxm1",
        "airport",
        "lp_degen2",
    ]
    .iter()
    .map(|m| root.join(format!("{m}.nl")))
    .collect()
}
