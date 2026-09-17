//! How many of POUNCE's KKT triplet slots are numerically zero — and are they
//! zero *always*, or only at iteration 0?
//!
//! This exists because the first cut of the assessment answered only half of
//! that. It measured `airport`'s **first** factorization, found 1768 of 2016
//! stored entries exactly zero, and wrote "POUNCE's KKT carries 88% explicit
//! zeros" — a claim about a solve, inferred from its first iteration.
//!
//! Across the whole solve the figure is nothing like that:
//!
//! ```text
//! model               slots  factzns  zero@it0  zero@last  zero-always
//! airport              2100       17     88.2%       2.0%         2.0%
//! eigena2              1825       43     88.5%       9.0%         0.0%
//! deb7                 8063      150     38.3%      11.1%        11.1%
//! convex_qp_qscfxm1    4395       25     24.2%       7.5%         7.5%
//! lp_degen2            5402      251      8.2%       0.0%         0.0%
//! wyndor_min             15       10     20.0%      20.0%        20.0%
//! ```
//!
//! Iteration 0 is not representative and could not be: at the starting point
//! much of the barrier diagonal and the Hessian contribution has no value yet,
//! while the *pattern* is fixed once by `initialize_structure` and reused for
//! every later factorization. A slot that is zero at iteration 0 and nonzero at
//! iteration 5 **must** be in the pattern. So the zeros are not waste — they
//! are the price of the fixed symbolic pattern, which is the thing that makes
//! refactorization cheap in the first place.
//!
//! `zero-always` is the only column that measures waste: slots that are zero in
//! every factorization of the solve. That is 0–11%, and 0% on two of six.
//!
//! The consequence for the cross-solver comparison is in `compare.rs`: RSLAB
//! drops numerically-zero entries when it materializes `L`, so its
//! factor-size figure moves from 88% to 2% across one solve for an unchanged
//! pattern. That makes it value-dependent and useless as a fill metric, which
//! is why the harness reports the structural size instead.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use pounce_algorithm::alg_builder::LinearSolverChoice;
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

struct Cap {
    inner: pounce_feral::FeralSolverInterface,
    sink: Arc<Mutex<Vec<Vec<Number>>>>,
    meta: Arc<Mutex<(Index, Vec<Index>, Vec<Index>)>>,
}
impl SparseSymLinearSolverInterface for Cap {
    fn initialize_structure(
        &mut self,
        d: Index,
        nz: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        *self.meta.lock().unwrap() = (d, ia.to_vec(), ja.to_vec());
        self.inner.initialize_structure(d, nz, ia, ja)
    }
    fn values_array_mut(&mut self) -> &mut [Number] {
        self.inner.values_array_mut()
    }
    fn multi_solve(
        &mut self,
        nm: bool,
        ia: &[Index],
        ja: &[Index],
        nr: Index,
        r: &mut [Number],
        c: bool,
        k: Index,
    ) -> ESymSolverStatus {
        if nm {
            self.sink
                .lock()
                .unwrap()
                .push(self.inner.values_array_mut().to_vec());
        }
        self.inner.multi_solve(nm, ia, ja, nr, r, c, k)
    }
    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
    }
    fn increase_quality(&mut self) -> bool {
        false
    }
    fn provides_inertia(&self) -> bool {
        true
    }
    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("crates/pounce-cli/tests/fixtures");
    println!(
        "{:<22} {:>6} {:>8} {:>9} {:>11} {:>11} {:>13} {:>13}",
        "model", "n", "slots", "factzns", "zero@it0", "zero@last", "zero-always", "never-used"
    );
    for name in [
        "airport",
        "eigena2",
        "deb7",
        "convex_qp_qscfxm1",
        "lp_degen2",
        "wyndor_min",
    ] {
        let path = root.join(format!("{name}.nl"));
        if !path.is_file() {
            continue;
        }
        let prob = match pounce_nl::nl_reader::read_nl_file(&path) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let tnlp: Rc<RefCell<dyn pounce_nlp::tnlp::TNLP>> =
            Rc::new(RefCell::new(pounce_nl::nl_reader::NlTnlp::new(prob)));
        let sink = Arc::new(Mutex::new(Vec::new()));
        let meta = Arc::new(Mutex::new((0, vec![], vec![])));
        let (s2, m2) = (Arc::clone(&sink), Arc::clone(&meta));
        let mut app = IpoptApplication::new();
        app.initialize().unwrap();
        app.initialize_with_options_str("print_level 0\nmax_iter 200\n")
            .unwrap();
        app.set_linear_backend_factory(Box::new(
            move |_c: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
                Box::new(Cap {
                    inner: pounce_feral::FeralSolverInterface::new(),
                    sink: Arc::clone(&s2),
                    meta: Arc::clone(&m2),
                })
            },
        ));
        let _ = app.optimize_tnlp(tnlp);

        let snaps = sink.lock().unwrap();
        let (n, _irn, _jcn) = meta.lock().unwrap().clone();
        if snaps.is_empty() {
            continue;
        }
        let slots = snaps[0].len();
        let z0 = snaps[0].iter().filter(|v| **v == 0.0).count();
        let zl = snaps.last().unwrap().iter().filter(|v| **v == 0.0).count();
        // A slot that is zero in EVERY captured factorization is the only kind
        // that is genuinely never carrying information.
        let mut always = vec![true; slots];
        for s in snaps.iter() {
            for (k, v) in s.iter().enumerate() {
                if *v != 0.0 {
                    always[k] = false;
                }
            }
        }
        let zalways = always.iter().filter(|b| **b).count();
        println!(
            "{:<22} {:>6} {:>8} {:>9} {:>10.1}% {:>10.1}% {:>12.1}% {:>12}",
            name,
            n,
            slots,
            snaps.len(),
            100.0 * z0 as f64 / slots as f64,
            100.0 * zl as f64 / slots as f64,
            100.0 * zalways as f64 / slots as f64,
            zalways
        );
    }
}
